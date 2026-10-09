# Chemeleon torch→Paddle 迁移工作量深度调研报告

调研日期：2026-10-09。调研性质：正式立项前的静态深调研——只读源码，未运行任何仓库代码、未安装依赖、未下载权重。调研对象：`/home/lkyu/baidu/Chemeleon/Chemeleon`（torch 原仓库，上游 main `df4c674` 的完整副本，62 文件 / 40 py / 5,437 行）；参照仓库：`/home/lkyu/baidu/PaddleMaterials`（ppmat，develop `e759a98`）。

方法：8 个维度（盘点/模型核心/文本路径/数据/训练/采样/依赖/硬件）并行深研 + 4 路对抗验证（文本路径、依赖面、硬件、模型核心），验证将初版报告的参数量算术错误（CSPNet 曾报 19.85M，实为 12.88M）、边数公式（n(n−1) → n² 含自环）、「4 ckpt 全作废」的夸大（实为 2 个 Chemeleon ckpt 的 FiLM 键失配）等逐一修正，本文全部采用修正后数字。引用约定：路径相对各自仓库根，`C:` 指 Chemeleon 仓、`P:` 指 PaddleMaterials 仓。无法静态确认的事项在第 12 节集中列出，正文涉及处以「未确认」标注。

本文只做分析与工作量判断，不做后续工作的硬性规划与决策。

---

## 1. 仓库全景

### 1.1 规模与构成

62 文件中 `data/` 占 162MB（CSV 数据），代码主体 40 个 py 共 5,437 行。按迁移视角分类（LOC 实测）：

| 分类 | 文件 | 小计 LOC |
|---|---|---|
| 模型前向核心 | cspnet.py(405)、scatter.py(112) | 517 |
| 扩散与损失 | diff_utils.py(329)、chemeleon.py 的 forward/损失段、base_module.py(135) | ~470 |
| 数据管线 | datamodule.py(94)、mp_dataset.py(66)、dataset_utils.py(64)、data_utils.py(438，其中约 390 行为死路径) | ~660 |
| 训练编排 | run.py(92)、run_crystal_clip.py(93)、config.py(312)、sweep.py(33，死) | 530 |
| 采样推理 | chemeleon.py 采样段、schema.py(88)、cli.py(203)、scripts/ 三个采样脚本(237) | ~620 |
| 文本与 CLIP | text_encoder.py(205)、crystal_clip.py(191)、text_encoder/\_\_init\_\_.py(17) | 413 |
| 评估 | evaluate.py(299)、optimize.py(128) | 427 |
| 应用与演示 | app/ 五文件(1,017)、client.py(17)、demo.ipynb、visualize.py(217) | ~1,256 |

去掉 app/演示层与数据溯源脚本后，核心可迁移面约 3,100 行；其中纯生成侧（不含 text_encoder/ 与 CLIP 入口）约 2,400 行。

### 1.2 死代码与坏死代码（迁移面直接扣减）

| 项 | 判定 | 证据 |
|---|---|---|
| knn 边路径整链 | 默认 `edge_style="fc"`（C:chemeleon/config.py:36）下不可达；且 `segment_coo/segment_csr` import 已注释（C:chemeleon/utils/data_utils.py:7）而调用仍在（data_utils.py:98,102,307,312,331,338,390），一旦执行即 NameError | cspnet.py:319-343 |
| data_utils.py:151-438 | radius_graph_pbc / get_max_neighbors_mask / decimal_to_bits 等，含上条死链与全仓无引用函数 | data_utils.py |
| cspnet.py 的 reorder_symmetric_edges / select_symmetric_edges / bincount | 仅 knn 分支使用 | cspnet.py:236-317,293 |
| sweep.py | 超参键名与 config 实际键失配（hidden_size/learning_rate vs hidden_dim/lr），注入不生效 | sweep.py:23-25 |
| client.py | Lightning 模板残留，协议不符 | client.py:16 |
| app/dash_app.py 本地模式 | 调用不存在的 `load_from_default_checkpoint` | dash_app.py:31 |
| config 死字段 | max_num_atoms、meta_stable_test、dynamic_stable_test、optimization_test、wandb_id、load_path 全仓无消费 | config.py:56,85,92-95 |
| scheduler 分支 | `constant_with_warmup` 未实现（ValueError）；`linear_decay` 的 LinearLR 传参错误 | base_module.py:117-128 |

### 1.3 入口与两阶段结构

- 第一阶段 CLIP 对比学习：`run_crystal_clip.py` → `CrystalClip`（BERT 文本塔可训 + CSPNet 图塔 + 双投影，C:chemeleon/text_encoder/crystal_clip.py:15-191）。
- 第二阶段扩散：`run.py` → `Chemeleon`，通过 `text_encoder="chemeleon/clip-*"`（wandb artifact，text_encoder.py:67-82）消费第一阶段产物。
- 推理：`Chemeleon.load_general_text_model / load_composition_model`（chemeleon.py:97-135）显式传 `path_ckpt_clip` 从本地 ckpt 组装 TextEncoder；四个 ckpt 的 figshare 直链在 C:chemeleon/constants.py:9-14。
- 配置体系：sacred（C:chemeleon/config.py:2-4，**requirements.txt 未声明的隐藏依赖**）+ 25 个 named_config；CLI 是 click（cli.py）与 fire（scripts/）双轨。

### 1.4 数据资产（随仓库完整提供）

`data/mp-40/` 共 165MB。**真实结构数：train 28,636 / val 3,181 / test 708**（CSV 的 cif 列是多行引号文本，物理行数 127 万不等于结构数；三集之和 32,525 恰等于 mp-total.csv 记录数，闭环验证）。15 列 schema：cif（P1 展开）、composition（字母序约化式）、crystal_system、prompt（GPT-3.5 生成的英文描述句，三 split 均无空值）等。数据筛选：num_sites≤40、energy_above_hull≤0.25、experimental、排除气相单原子与 abc>20 结构；test 为 `created_at ≥ 2018-08-04` 的时序 holdout（data_preparation.ipynb cell 11-20）。原子数分布：train 均值 17.27 / max 40，test 均值 26.68；36-40 原子记录占 9%。

---

## 2. 模型架构（生成主干）

### 2.1 结构总览

```
batch(atom_types[B_n], frac_coords[B_n,3], lattices[B,3,3], text[List[str]])
  │ 三路并行加噪（同一个 t，按 batch 采样后广播到节点）
  ├─ 类型：D3PM 吸收态离散扩散（吸收到 index 0 = dummy 原子）
  │    q_sample = Gumbel-max argmax(log(x0·Q_{1..t}) + gumbel)   → x_t 为 [B_n] long 索引
  ├─ 晶格：VP/DDPM（cosine β，eps 预测），仅对 7 个自由元加噪
  │    mask = [[1,0,1],[1,1,1],[0,0,1]]（chemeleon.py:70-73）
  └─ 坐标：VE wrapped-normal（σ: 0.01→1.0 对数表，%1.0 回卷）
       回归目标 d_log_p_wrapped_normal(σε,σ)/√σ_norm（21 项周期和，diff_utils.py:35-46）
  │
  ├─ 文本（可选）：BERT CLS → CLIP text_proj → CFG null 替换(768 维) → text_emb MLP(768→512)
  ▼
CSPNet：节点 Embedding(104,512) → [FiLM(t128,text512) → CSPLayer]×6 → 三头输出
  FiLM：mlp_cond Linear(640→1024) 出 scale/shift，乘性调制（无 cross-attention/AdaLN）
  CSPLayer：edge_mlp Linear(1801→512→512)（输入 = h_i,h_j,内积矩阵 9 维,Fourier 坐标差 768 维）
            + scatter_mean 聚合 + node_mlp Linear(1024→512→512) + 残差
  输出头：coord Linear(512→3)、lattice Linear(512→9)→einsum 乘当前晶格、type Linear(512→104)
```

- 边构造在模型 forward 内：fc 全连接（每图 n² 条**含 n 条自环**——`dense_to_sparse` 保留对角，cspnet.py:319-324；PyG block_diag 路径）；边特征为分数坐标差 `(x_j−x_i)%1.0` 的 128 频率 Fourier 展开（768 维）。
- 头部注释即标明源自 DiffCSP（cspnet.py:1、diff_utils.py:1）。
- 损失：`loss = 1.0·(VB_KL + CE) + 1.0·lattice_masked_MSE + 1.0·coords_MSE`（chemeleon.py:211-235；权重 config.py:57-59 全 1.0，d3pm_hybrid_coeff=1.0）。

### 2.2 与 ppmat DiffCSP 的同源性（工作量折减的核心依据）

ppmat 已有 `ppmat/models/diffcsp/diffcsp.py`（RuntimeMixin 组织、`denoise_step` 边界、forward 内嵌 loss_dict、sample()），其 CSPLayer/SinusoidsEmbedding/fc 建边/wrapped-normal 数学与 Chemeleon **逐行同构**（P:diffcsp.py:46-172,283-295 ↔ C:cspnet.py:38-181,319-324），且 diffcsp_mp20.yaml（P:structure_generation/configs/diffcsp/diffcsp_mp20.yaml:63-70）在 hidden 512 / 6 层 / num_freqs 128 / fc / ln 的**同规模**下配置过（batch 256 单卡，yaml:128）。需要为 Chemeleon 新写/改造的增量：

| 增量 | 规模估计 | 说明 |
|---|---|---|
| FilmLayer | ~45 行 | ppmat DiffCSP 用 concat+Linear 一次注入（P:diffcsp.py:341），Chemeleon 是每层 FiLM，权重不通用（C:cspnet.py:55-97） |
| Chemeleon 语义的 D3PM | ~180 行 | ppmat 的 D3PMScheduler（P:ppmat/schedulers/scheduling_d3pm.py:26）数学同构但实现谱系不同：解析式 absorbing 到 dim-1、仅 standard β、Categorical.sample；Chemeleon 是显式 Q 矩阵连乘 + cosine β + absorbing 到 index 0 + Gumbel argmax + logits-KL。不可直接互换 |
| 晶格 7 元素 mask 噪声协议 | ~30 行 | ppmat 全库无 lattice mask（grep 证实） |
| CFG 双前向 + null embedding | ~50 行 | ppmat 侧 MatterGen 的 p_unconditional 机制同构可参照（P:ppmat/models/mattergen/mattergen.py:2588,2627-2700） |
| 三路联合 loss 与 PC 采样循环 | ~200 行 | 大量抄 chemeleon.py:379-467 的结构 + ppmat scheduler API |
| type_out 头 | 0 | ppmat diffcsp 的 pred_type 分支已预留（P:diffcsp.py:268-271） |

骨干代码估计 60-70% 可直接复用 ppmat DiffCSP 现成实现。

### 2.3 参数量（验证后数字，手算复核）

| 组件 | 参数量 | 出处 |
|---|---|---|
| CSPNet（默认 text_guide 配置） | **12,876,392 ≈ 12.88M（fp32 ≈ 51.5MB）** | node_emb 53,248 + FiLM 920,064 + 6×CSPLayer 1,973,760 + 输出头 60,520（cspnet.py:113-127,210,225-233） |
| text_emb MLP + null_text_embeds | 986,624 ≈ 0.99M | text_encoder.py:40-46 |
| 可训练合计（默认 BERT 路径） | **13.86M ≈ 55MB** | |
| 可训练合计（CLIP 编码器路径，即发布 ckpt 形态） | ≈ **27.7M** | 经 path_ckpt_clip/artifact 加载时 clip_model.graph_encoder(11.96M)+投影(1.84M) 未被冻结且前向未使用（text_encoder.py:48-51,119-125 的冻结只作用于 BERT） |
| BERT（MatTPUSciBERT，BERT-base） | ≈ 110M ≈ 440MB（词表 31,090，未确认） | text_encoder.py:113 |
| unguided（text_guide=False） | CSPNet = 12,352,104 ≈ 12.35M | FiLM 不删除、收缩为 time-only Linear(128→1024)（cspnet.py:211-212 + chemeleon.py:78） |
| CrystalClip 全体 | ≈ 123.8M（图塔 11.96M + 双投影 1.84M + BERT 110M），全参可训 | crystal_clip.py:28-73,32 |
| D3PM buffer | 2 × (1001,104,104) fp32 = **86.6MB**（进 state_dict） | diff_utils.py:168-191 |

**checkpoint 构成判定**：chemeleon ckpt 723MB ≈ 冻结 BERT 440MB + 可训练 55MB + Adam 状态 111MB（CLIP 路径则为 222MB）+ D3PM buffer 86.6MB ≈ 691-750MB，量级自洽（仓内无官方体积标称；723MB/1.4GB 为用户手动下载实测值）。**723MB ≠ 1.8 亿可训练参数——扩散主干仅 12.88M，大头是冻结文本编码器与 buffer**。CLIP ckpt 1.4GB ≈ 123.8M 全参权重 495MB + Adam 990MB ≈ 1.49GB，自洽。双 BERT 键共享 storage 的去重行为未下载验证（未确认）。

---

## 3. 文本条件路径专章（「剪不剪」的全部结构性事实）

### 3.1 composition 条件就是文本条件

全仓库唯一条件入口是 `batch.text → TextEncoder.get_text_embeds`（chemeleon.py:192-195）；**不存在独立的 composition one-hot/查表编码**。`text_targets=["composition"]` 时 dataset 把 CSV composition 列的字符串（如 `"Ho5 Mo2 O12"`）直接作为文本（mp_dataset.py:53-63）。链路：文本 str → BertTokenizer（padding="longest", max_len=256）→ BERT CLS（text_encoder.py:174-177）→ CLIP text_proj（:180-182）→ **CFG null 替换发生在 768 维、投影前**（:186-202）→ text_emb MLP 768→512 → FiLM（repeat_interleave 到节点级，cspnet.py:364-367）。

推理侧同样以文本喂 composition：`text_input = comp` 后 `model.sample(...)`（sample_target_composition.py:41-47）。**两个发布形态（general_text / composition）的采样代码完全共用**，区别只在 ckpt 与字符串内容。

### 3.2 注入接口与剪裁的结构性判据

- FiLM 是 time 与 text 的**同一入口**：`mlp_cond = Linear(time_dim+text_dim, hidden×2)`，即 `film_layer.mlp_cond.0.weight` 形状 `[1024, 640]`（cspnet.py:70-73）。`text_guide=False` 时传 `text_dim=0`（chemeleon.py:78），该层变 `[1024,128]`——**这是剪文本时唯一严格失配的权重层**。
- 仓库原生预留了 `unguided` named_config（config.py:138-143），torch 侧零改动即可训 unguided（但仍需 pip 装 transformers/wandb，因 chemeleon.py:27-28 模块级 import；仅免在线）。官方姊妹仓库 chemeleon-dng 即「同架构去文本编码器、256 步」的先例（README.md:38）。

### 3.3 四个 checkpoint × 路线的权重可用性矩阵

| checkpoint | 剪（text_dim=0） | 中间路线（保留 640 维 FiLM + 固定 null 条件） | 不剪（完整结构） |
|---|---|---|---|
| chemeleon_composition (723MB) | 部分：除 film_layer.mlp_cond 外可 strict=False 加载 | **decoder 100% 可载**（保留 text_emb+null 共 ~0.99M 参数位） | 可用（权重转换后） |
| chemeleon_general_text (723MB) | 同上 | 同上 | 可用 |
| clip_composition / clip_general_text (各 1.4GB) | 本就与扩散 decoder 无关（其 CSPNet 为 time_dim=0/text_dim=0 的 CLIP 专属形态） | 同左 | 可用（BERT+text_proj 初值，仅复训一阶段需要） |

注意：**不剪路线下其实只需 chemeleon-*.ckpt 单文件**（其中已含 BERT/text_proj/text_emb/null），CLIP ckpt 仅一阶段复训需要。

### 3.4 三条路线的工作量分解（按 HIENet 七阶段路径映射，人日为量级估计）

**路线 A：硬剪（text_dim=0，unguided）**

| 阶段 | 内容 | 估计 |
|---|---|---|
| 环境 | paddle-gpu（WSL2）+ ppmat；无 torch 生态 vendor | 0.5 天 |
| 数据 | MP40Dataset（仿 mp20_dataset.py，CIF 解析+niggli+canonical 三步与 ppmat BuildStructure 逐语义一致） | 1 天 |
| torch golden | `run.py with unguided` 小规模跑通 + 固定 batch 顺序 golden | 1-2 天 |
| vendor | **零**（见第 8 节） | 0 |
| paconvert+手修 | diff_utils 329 + cspnet 405 + chemeleon.py 去 text + schema 88；scatter/data_utils 复用 ppmat | 3-5 天 |
| 前向对齐 | fp64 golden，判据 1e-5；重点 D3PM 连乘/wrapped normal/FiLM 顺序 | 2-3 天 |
| 训练对齐 | 三项 loss 逐步对拍 mean_rel ≤0.2% | 2-3 天 |
| ckpt 转换 | chemeleon-*.ckpt → pdparams（decoder 非 FiLM 部分 + buffers） | 1-2 天 |
| ppmat 契约 | forward 双开关 + loss 内嵌 + sample() + RuntimeMixin denoise_step + yaml + 注册 | 3-4 天 |
| **合计** | | **≈ 14-21 人日** |

剪的删改面：核心包 553 行整删（text_encoder.py 205 + crystal_clip.py 191 + \_\_init\_\_ 17 + run_crystal_clip.py 93 + sample_prompt.py 47）+ 外围 1,309 行（app 1,017 + client 17 + generate_text_prompt 85 + 两个 composition 采样脚本 190）+ 约 300 行修改。transformers/einops 随剪**整体消失**（全仓仅 text_encoder.py:7-14、crystal_clip.py:8 两处 import；einops 仅 text_encoder.py 1 处）。

**路线 A′：软剪（接口留缝）——保留 FiLM(640) 结构与 text_emb/null 参数位，恒以 null 条件推理/训练**：相对路线 A 增加极少工作量（不做删除，只固定输入），换来 chemeleon-*.ckpt 的 decoder 100% 可载、后续文本能力可增量恢复。代价：模型语义是「恒 null 条件」（null 在训练分布内，cond_drop_prob=0.2），与原版 unguided 训练分布不同，生成质量需小规模评测对比（未评估）。

**路线 B：不剪（text-guided 完整版）**：路线 A 基础上追加——

| 追加项 | 估计 |
|---|---|
| BERT 权重 HF→paddle 转换 + BertModel/BertTokenizer paddle 侧落地（paddlenlp 或手写 ~400 行；tokenizer 可用无 torch 的 `tokenizers` 库） | 2-3 天 |
| TextEncoder 迁移（砍 T5/Llama/Phi 实验分支后 ~120 行）+ CFG 采样分支 | 1-2 天 |
| CrystalClip 一阶段（可选后置；推理用 ckpt 不需要） | 2-3 天（后置则 0） |
| 4 ckpt 全量转换（含 BERT 子树） | 1 天 |
| 文本路径前向/训练对拍 | 1-2 天 |
| **合计** | **+7-11 人日 ≈ 21-32 人日** |

另有一条**训练侧折衷**：BERT 冻结（config.py:49）且训练集文本集合静态（train 28,636 条）→ 文本 embedding 可离线预编码为 [N,768] 缓存，训练侧无 BERT/transformers；**推理侧仍需 paddle BERT**（任意新文本现算）。注意两个 caveat：(a) null 替换在 768 维、null_text_embeds 是可学习参数，预编码不能覆盖 null 路径（每步仍需对 1 个向量过 text_emb MLP，代价可忽略）；(b) 若原训练期 BERT 的 dropout 实际处于开启状态（PL 的 model.train() 会递归翻转构造时的 .eval()，见第 10 节风险 8），则离线预编码（eval 模式）与原训练语义有差异，需实测确认。

---

## 4. 数据管线

- **无任何归一化统计量**：lattice 原值（Å 量纲）直接进 VP 加噪，frac_coords 天然 [0,1)，原子类型即原子序数（1..103，0 为 dummy）。notebook 里的 lattice_params.txt 未随仓库分发且无代码读取。**不应引入 ppmat 的晶格归一化变换，否则破坏与官方 ckpt 的数值对齐**。
- Dataset 链：`pd.read_csv` 一次入内存 → 每 `__getitem__` 现场 `Structure.from_str(cif)` + Niggli 约化 + canonical 重建 + 转 ASE → 张量化（mp_dataset.py:38-66）。**无缓存**，num_workers=0（config.py:24）→ 每 epoch 重复解析 28,636 个 CIF，是训练吞吐的隐性 CPU 瓶颈（GPU 空转风险）；ppmat 的 pkl 结构缓存（P:ppmat/datasets/mp20_dataset.py:249-262）可一次构建终身复用。
- batch 组装：PyG 扁平拼接，无 padding；`max_atoms=104` 是**词表大小**而非原子数上限（`max_num_atoms=50` 是死配置）。
- PyG 消费面 = **4 个符号**：`Data`、`Batch`（from_data_list/to_data_list/num_graphs/属性访问）、`DataLoader`（仅 batch_size/shuffle/num_workers/pin_memory 四参数）、`dense_to_sparse`（fc 建边，唯一算子调用，cspnet.py:323）。ppmat 的 geometric_data_type（5 文件 1,058 LOC，HIENet 迁移时已踩平）全覆盖；fc 建边直接抄 P:diffcsp.py:283-295 的纯 paddle 写法（顺带绕开 torch.block_diag）。
- ppmat 覆盖度：CSV 读取/CIF 解析+Niggli+canonical/结构缓存/collate/scatter/调度器/采样外壳全部现成；**缺**的只有文本编码路径（BERT/tokenizer）与 MACE 评估。数据侧净新增 = 1 个 MP40Dataset 类（0.5-1 天）。
- 索引约定差异要注意：ppmat diffcsp 用 `atom_types-1`（P:diffcsp.py:526），Chemeleon 用原始 Z 且词表 104 含 dummy 0——不统一会导致 D3PM 转移矩阵与 Embedding 权重错位。

---

## 5. 训练编排

- LightningModule 消费面（替换 ppmat BaseTrainer 的核对清单）：`save_hyperparameters`×3、`self.log`×8、`load_from_checkpoint`×5、`trainer.max_steps`、`seed_everything`×3、Trainer 9 个 flag、三个 callbacks（ModelCheckpoint/LearningRateMonitor/EarlyStopping）、WandbLogger。**全部是脚手架，不进数值内核**。
- ppmat BaseTrainer 零改动覆盖：fit 循环、每 epoch 验证、Adam、ReduceLROnPlateau（indicator/factor/patience/min_lr 全对得上，P:ppmat/optimizer/lr_scheduler.py:727-767）、梯度裁剪 global-norm 0.5（P:ppmat/optimizer/\_\_init\_\_.py:91-102）、周期 ckpt、resume、wandb/visualdl 日志、seed 三家齐种（P:ppmat/utils/misc.py:483-491）。唯二缺口：EarlyStopping（可舍弃或自写 ~20 行）与 save_last 语义。
- **训练 RNG 消费序列**（对拍时需逐步固定）：① `np.random.choice` 采 t（diff_utils.py:104-106，**numpy 全局 RNG，非 torch**）→ ② `torch.rand` D3PM Gumbel（chemeleon.py:164）→ ③ `torch.randn_like` 晶格（:174）→ ④ `torch.randn_like` 坐标（:179）→ ⑤ 条件丢弃 `uniform_`（text_encoder.py:195-197）→ ⑥ shuffle。**初始化期**：SigmaScheduler 构造消耗 10000×1000 个 randn 算 sigmas_norm buffer（diff_utils.py:49-54）+ null_text_embeds 的 randn（text_encoder.py:46）——**构造顺序改变会平移全局 RNG 流**。
- 训练配置底稿（ppmat yaml 忠于原作者的核对基础，config.py:7-95）：batch 128、Adam lr 1e-3 wd 0、reduce_on_plateau(factor 0.8, patience 200, min_lr 1e-6)、gradient_clip 0.5、max_epochs 1000、early_stopping 300、seed 0、三路 loss 权重 1:1:1、cond_drop_prob 0.2、timesteps 1000、cosine β。
- 每训练 step：decoder 前向 1 次 + 反向；224 step/epoch（28,636/128）。

---

## 6. 推理/采样

- 主循环：严格 1000→1 逐步 ancestral 反向（chemeleon.py:379），**无 DDIM/少步采样选项**（无 set_timesteps/num_inference_steps 接口）；ppmat DiffCSP 反而支持 num_inference_steps（P:diffcsp.py:545,556-557）。
- 每步 2 次 model_predictions（Predictor :388 + Corrector :440）；text_guide 下每次内部 CFG 双前向 → **每步 4 次 decoder 前向，一次 sample() = 4000 次**；unguided 2 次/步。文本编码仅在循环外 2 次（cond/null，:364-374）。t=1 时所有随机项置零。
- 三路反向：类型 = p_logits 后验 + Gumbel argmax（t=1 直接 argmax 预测 logits）；晶格 = DDPM 后验均值+方差采样，全程乘 7 元素 mask，首步 clip(-6,6)；坐标 = wrapped score 的 PC 采样（Predictor 步长 σ² 差分 + Langevin Corrector step_lr=1e-5）。
- 后处理：`TrajectoryContainer.get_atoms`（schema.py:57-83）每步把张量转 ASE Atoms（>103 类型压 0、排序）；非 stream 模式 `list()` 收集全部 1000 步（chemeleon.py:483-486）→ CPU 内存与时间隐性开销。有效性检查（abc<60Å、最小原子距>0.5Å、组成匹配）与去重（StructureMatcher）都在脚本层。
- 评估：test 708 条 × 每条 20 样本（evaluate.py:23,112-117），7 类指标（valid/unique/SM/成分/晶系/格子系 + MACE 亚稳）；**无 CDVAE 式 COV 覆盖率指标**。全量评估 ≈ 12-24h GPU + MACE（torch 生态，建议留独立 torch 容器或暂缺省）。

---

## 7. 硬件开销（RTX 4060 Ti 16GB 单卡 + WSL2，静态估算）

### 7.1 训练

- 默认 batch 128 → 平均 2,211 节点 / 52,672 条 fc 边（含自环；n² 公式）。每层 edge_mlp 首层输入 [E,1801] ≈ 379MB，6 层激活 ≈ 3.7GB；加 FiLM/D3PM/持久驻留（参数 55MB + 梯度 + Adam 111MB + BERT 440MB + buffer 86.6MB）→ **典型峰值 ≈ 5.4-6.0GB，16GB 富余**。理论极端（整批全 40 原子，E=204,800）≈ 14GB 会 OOM，但随机 shuffle 下概率≈0。
- **可行包络：B=128-256（混合 ≤40 原子）稳过；B=512 属边界**。ppmat DiffCSP 同规模网络单卡 batch 256 训练佐证。
- D3PM 无显存炸弹：转移矩阵始终按 [B_n,104,104] 查表（B=128 时 95.6MB×2 次/forward）。但存在一个**性能坑**：`q_one_step_transposed` 未注册为 buffer（diff_utils.py:193），`Module.to()` 不搬移 → 每训练步 2×43.3MB CPU→GPU 拷贝（约 7-10ms），推理每 timestep 1 次（1000 步累计 ~43GB H2D ≈ 4-6s）。迁移时可顺手修（注册 buffer）。
- 训练时间：224 step/epoch；GPU 界 0.45-0.6 s/step（HIENet 同卡锚点外推）→ epoch ≈ 2 分钟、千 epoch 28-37h；**CPU 界**（num_workers=0 + 每 epoch CIF 重解析，估 5-15ms/结构）→ 1-2 s/step、epoch 4-8 分钟。对齐工作流（固定权重逐步 loss 对拍）百步级，分钟-小时，无压力。
- CLIP 一阶段（若做）：全参 123.8M，峰值 ≈ 6-8GB，16GB 可行。
- 模型无任何二阶梯度需求（损失仅 MSE/CE/KL）——历史「二阶梯度大结构 OOM」教训不适用；WSL2 cgroup 封顶纪律仍适用。

### 7.2 推理

- 显存：模型+buffer ≈ 0.64GB（含 CLIP 图塔 55MB）+ 单次前向激活（20 样本×40 原子 E=32,000 → edge cat 232MB）→ **峰值 <1.6GB，无炸弹**。
- 时间：4000 次小前向（5-15ms/次，kernel 启动开销主导）+ 每步 43MB H2D + 每步 ASE 后处理 → 单次 sample()（20 样本、≤40 原子、CFG）**≈ 0.7-2 分钟量级**；unguided 减半。

### 7.3 文本 vs 剪裁的硬件量化

| 项 | 文本引导（现状） | unguided（硬剪） |
|---|---|---|
| 常驻显存 | ≈ 0.58GB（BERT 440MB 为主） | ≈ 0.14GB |
| 可训练参数 | 13.9M（CLIP 路径 27.7M） | 12.35M |
| 训练峰值（B=128） | ~5.4-6.0GB | ~4.6-5.1GB |
| 推理前向/步 | 4（CFG） | 2 |
| 单次 sample() | ~0.7-2 min | ~0.35-1 min |
| ckpt 体积 | ~723MB | ~140MB 纯权重 / ~240MB 含 optimizer |

**结论：文本路径的硬件代价集中在 BERT 常驻 440MB + 推理 CFG 双倍前向 + ckpt 体积；训练算力占比 <10%。16GB 卡在两种形态下都可行——硬件不是剪/不剪的决定因素，决定因素是耦合度与权重兼容性。**

---

## 8. 依赖面与 torch 间接依赖（用户重点问题）

### 8.1 四桶分类（穷尽 grep，对抗验证确认无遗漏）

**桶 B torch 本体**：13 个文件；符号面约 60 个唯一符号，其中**非平凡映射仅 3+1 处**：`torch.block_diag`（cspnet.py:322；paddle ≥2.6 有 `paddle.linalg.block_diag`，且建议直接用 ppmat 的 fc 建边绕开）、`torch.bincount`（knn 死分支内）、`torch.div(rounding_mode="floor")`（data_utils knn 分支 + scatter.py:111，随文件整体替换）。

**桶 C 依赖 torch 的第三方（6+1 个）——逐项处置**：

| 依赖 | 使用面 | 深度 | 处置 | paddle 侧是否间接依赖 torch | 工作量 |
|---|---|---|---|---|---|
| torch-geometric | 4 符号（第 4 节） | 数据管线+前向+采样 | 换 ppmat geometric_data_type + diffcsp fc 建边；**零 vendor**（若 vendor 需 ~1,700 行，收益为负） | **否** | 0.5-1 天 |
| pytorch-lightning | 2 LightningModule + 1 DataModule + Trainer 全套 | 仅训练脚手架 | 换 ppmat BaseTrainer（覆盖度见第 5 节） | **否** | 1-2 天 |
| torchmetrics | 1 符号（MeanAbsoluteError，base_module.py:8） | 仅日志 | `paddle.metric` 或手写 | **否** | <10 行 |
| transformers | BertModel/BertTokenizer 等，2 文件 | **唯一进前向+采样的 torch 第三方** | 剪→整体消失；不剪→paddlenlp/手写 BERT + 权重转换（HF→paddle BERT 键名基本同名，100-200 行脚本） | 剪否 / 不剪由 paddle 侧 BERT 替代，transformers 本体仍不进 | 剪 0 / 不剪 1-2 天 |
| einops | rearrange ×3（text_encoder.py 1 文件） | 浅 | ppmat 已含 einops==0.8.0 且原生支持 paddle 后端；或 3 行改 unsqueeze | **否** | ≈0 |
| wandb | 日志 + artifact 下载 | 外围 | ppmat 已含 wandb==0.18.3（use_wandb 开关）；artifact 路径改 figshare 直链（constants.py 已有 URL） | **否**（wandb 本身无 torch 依赖，METADATA 核实） | 0.5 天 |
| mace（评估脚本） | evaluate.py:13,91 / optimize.py:13 | 仅评估 | 留独立 torch 评估容器或该指标暂缺省 | 训推**否** | 0（后置） |

**桶 D 纯 python 第三方**：pymatgen/ase/smact/numpy/pandas/pydantic/tqdm/plotly——逐一 grep site-packages 确认**零隐藏 torch 引用**，ppmat 同款依赖（pymatgen 2024.10.29 / ase / SMACT 2.5.5），原样保留。

**隐藏依赖**（requirements.txt 未声明）：sacred（config.py:2，换 hydra/omegaconf）、litserve 与 dash_bootstrap_components（app 层，随删）、plotly、requests、mace、numpy/pandas（隐式）。requirements 声明但未用：jupyterlab、smact 重复声明两次。

### 8.2 paddle 侧最终运行时依赖

- **训练必需**：paddlepaddle-gpu、numpy、pandas、pymatgen、ase、omegaconf/hydra-core、tqdm + ppmat 内部设施（scatter/geometric_data_type/trainer/scheduler）。einops 可选。
- **仅推理追加**：（若保留文本）paddle 版 BERT 权重 + tokenizer（tokenizers 库无 torch）；requests（figshare 下载）。
- **仅评估**：smact（ppmat 已有）；mace + torch（独立容器）。
- **删除**：sacred、dash、streamlit、litserve、langchain 系、jupyterlab、click/fire 收敛为 ppmat 惯例入口。

**明确不进 paddle 运行时的 torch 系依赖：torch、torch-geometric、torch_scatter（本就零引用）、pytorch-lightning、torchmetrics、transformers（剪）或不进本体（不剪）、mace。用户担心的「面向 paddle 项目间接依赖 torch」问题：全部 torch 系第三方均有零 torch 的替代路径，无一需要 vendor。**

### 8.3 与 HIENet 依赖工作量对比

HIENet 最终形态 vendor 了 e3nn 40 文件 10,210 行 + nn 13 文件 2,079 行（合计 53 文件 12,289 LOC）+ torch_scatter paddle stub。**Chemeleon 对应工作量：零 vendor 文件、零 C 扩展对拍**——无 e3nn（CSPNet 是普通 MLP 消息传递）、无 torch_scatter（仓库自带 112 行纯 Python scatter，ppmat 有同款）、PyG 4 符号被 ppmat 全覆盖。迁移成本从「vendor 对拍」转移到两处模型本体改写（FiLM ~45 行 + Chemeleon 语义 D3PM ~180 行）+ ckpt 权重转换脚本（约 250-500 行）。

---

## 9. ppmat 可复用设施对照（file:line）

| 设施 | 位置 | 复用度 |
|---|---|---|
| scatter/scatter_mean | P:ppmat/utils/scatter.py:218,235,244 | 直接替换 |
| Data/Batch/Dataset | P:ppmat/datasets/geometric_data_type/（5 文件 1,058 行） | 全覆盖 4 符号 |
| CSPNet paddle 版 | P:ppmat/models/diffcsp/diffcsp.py:61-375 | 骨架高复用；注入方式不同需加 FiLM |
| fc 建边 | P:diffcsp.py:283-295 | 直接抄 |
| d_log_p_wrapped_normal | P:diffcsp.py:31-43 | 逐行等价 |
| 晶格 VP 调度 | P:ppmat/schedulers/scheduling_ddpm.py（squaredcos 同公式；step:462、add_noise:588） | 数学等价，需加 7 元素 mask |
| 坐标 VE wrapped 调度 | P:ppmat/schedulers/scheduling_sde_ve.py:401-549 | 同构（step_size=snr·(σ/σ₀)² ↔ step_lr） |
| D3PM 调度 | P:ppmat/schedulers/scheduling_d3pm.py:26-481 | 数学同构实现不同，需扩展（cosine β + absorbing 位置 0 + Gumbel）或按 Chemeleon 语义重写 ~180 行 |
| time embedding | P:ppmat/models/common/time_embedding.py:48 | 直接用 |
| StructureSampler | P:ppmat/sampler/structure_sampler.py:35（by_num_atoms:192 / by_condition:231） | 剪→by_num_atoms；不剪→by_condition |
| MatterGen 条件/CFG 先例 | P:ppmat/models/mattergen/mattergen.py:2546-2700、property_embeddings.py:203-216 | 同构机制参照 |
| RuntimeMixin/denoise_step | P:ppmat/models/common/runtime.py:100,119；先例 diffcsp.py:445、mattergen.py:2253,2612 | 契约直接套（扩散模型边界=denoise_step） |
| #310 forward 契约 | P:ppmat/models/hienet/hienet.py:91-113（本项目 HIENet 交付先例） | 范例 |
| BaseTrainer | P:ppmat/trainer/base_trainer.py | 零改动覆盖（除 EarlyStopping） |
| MP20Dataset 模式 | P:ppmat/datasets/mp20_dataset.py:42-166（含结构缓存） | 抄改即 MP40Dataset |

ppmat 全仓零 NLP 依赖（grep paddlenlp/transformers 无命中）——文本路线的 BERT 是 ppmat 侧净新增（也是贡献点：首个文本条件晶体生成模型）。

---

## 10. 对齐风险热点（按 HIENet 经验映射）

1. **sigmas_norm 是蒙特卡洛 buffer**（diff_utils.py:49-54）：构造期消耗 10000×1000 个 randn，两框架 RNG 流必然不同 → **必须从 torch golden ckpt 导出该 buffer，不能重算**（ppmat 同构函数同病，P:scheduling_sde_ve.py:43-50）。
2. **q_mats 1000 次 104×104 连乘**（diff_utils.py:178-184）：跨框架舍入累积漂移，同样从 ckpt 拷 buffer。
3. **D3PM Gumbel RNG**：对拍需两侧注入同一份固定 u 张量（或参数化 q_sample 的 noise 入参）。
4. **scatter 归约顺序**：scatter_add_ vs index_add 的 ulp 差经 6 层放大，fp64 golden + 1e-5 判据。
5. **负索引 `q_mats[t−2]`**（diff_utils.py:280）：t=1 回绕后由 where 丢弃，paddle 负索引 gather 语义需单测。
6. **`%1.0` 浮点取模**：wrapped score 的指数会放大 fp32 边界舍入差，golden 用 fp64。
7. **FiLM 的 chunk/cat 顺序**（cspnet.py:81,88）：错了不报错但数值全偏。
8. **冻结 BERT 的 train/eval 模式**：构造时 .eval() 但 PL 每 epoch 的 model.train() 会递归翻转 → 训练期文本嵌入可能带 dropout 随机性（PL 标准行为推断，未确认）；影响「逐步 loss 对拍」与「离线预编码」两条路，需 torch 侧实测确认后显式固定。
9. **t 采样的 numpy RNG 与噪声的 torch RNG 混用**：paddle 侧三家种子齐种（ppmat misc.set_random_seed 正是此设计），对拍时固定噪声张量注入。
10. **fc 边实现替换后的边序差**：block_diag+dense_to_sparse → cumsum+meshgrid 后边排列不同；scatter 和可交换不受影响，但 frac_diff/edge2graph 必须与 edges 一一对应，需排序后对拍边集合。
11. **ddp_find_unused_parameters_true（run.py:83 硬编码）的真实成因**：CLIP 路径下 graph_encoder/proj 可训练但前向未使用（非冻结参数本身需要）——单卡训练可忽略，迁移时须知。
12. **索引约定**：Chemeleon 原始 Z 词表（104 含 dummy 0）vs ppmat diffcsp 的 Z−1 约定。

---

## 11. 工作量综合评估

以 HIENet 七阶段路径为标尺的总量级（含上述全部验证后事实，人日为量级判断非承诺）：

| 阶段 | Chemeleon 预估 | 相对 HIENet 的差异 |
|---|---|---|
| 环境 | 0.5 天 | 相当 |
| 数据 | 1 天（数据随仓库、无归一化、ppmat 缓存现成） | 显著轻于 HIENet（彼时需伪标签/外部数据） |
| torch golden | 1-2 天 | 相当；注意 CLIP 路径的 wandb 在线依赖（unguided 可避开在线但避不开安装） |
| vendor | **0** | HIENet 为 53 文件 12,289 行——本模型最大减项 |
| paconvert+手修 | 3-5 天（核心 ~2,400-3,100 行，60-70% 骨架同源于 ppmat diffcsp） | 代码量与 HIENet 相当但同源折减明显 |
| 前向对齐 | 2-3 天 | 相当；风险点从等变张量转为 D3PM/wrapped 数学与 buffer 搬运 |
| 训练对齐 | 2-3 天 | 相当；无二阶梯度，难度低于 HIENet 的 force/stress |
| ckpt 转换 | 1-2 天（路线 A′/B 下 decoder 全量可载） | 相当 |
| ppmat 契约 | 3-4 天 | 相当（denoise_step 先例现成） |
| 文本增量（仅路线 B） | +7-11 人日 | HIENet 无对应物 |
| **合计** | **路线 A ≈ 14-21 / A′ ≈ A / B ≈ 21-32 人日** | HIENet 全程约 5-6 周（含大量探索），本模型路径成熟度更高 |

三个用户关注问题的最终回答：

1. **文本/生成工作量与剪不剪**：生成侧核心 ~2,400 行（其中 60-70% 与 ppmat diffcsp 同源）；文本侧收敛在 413 行两文件 + FiLM 注入点，且 composition 条件本身就是文本——**不存在「只留 composition、剪掉文本编码器」的中间形态**，可选项只有：硬剪（unguided，丢失 2 个 ckpt 的 FiLM 权重兼容）、软剪（保留 640 维 FiLM 恒 null 条件，decoder 权重 100% 可载）、完整保留（+7-11 人日，4 ckpt 全量可用）。剪/不剪的判定依据是权重复用与产品诉求（文本引导是论文核心卖点），硬件不构成约束。
2. **训练/推理硬件开销**：训练 B=128 峰值 5.4-6.0GB、16GB 富余（可行包络 B=128-256）；推理 <1.6GB。时间上训练 GPU 界 0.45-0.6 s/step（224 step/epoch，对齐工作流分钟-小时级），采样单批 0.7-2 分钟（CFG 4000 前向），全量评估 12-24h（可子集化）。CPU 侧瓶颈（每 epoch CIF 重解析）有 ppmat 缓存现成解法。
3. **torch 第三方依赖**：6+1 个（PyG/PL/torchmetrics/transformers/einops/wandb/mace），**全部有不引入 torch 的处置路径，零 vendor、零 C 扩展**；处置工作量合计约 2-4 人日（不含路线 B 的 BERT 增量）。纯 python 依赖干净，paddle 侧最终依赖表见 8.2。

---

## 12. 未确认项清单（需下载/装机/运行后才能关闭）

1. ckpt 内部键构成（双 BERT storage 去重、optimizer state 范围、CLIP 子树是否完整）——下载后 `torch.load` 验证。
2. MatTPUSciBERT 词表与精确参数量（31,090 词表为 SciBERT 值推断）。
3. PaddleNLP 是否有可直接复用的 BertModel 权重格式、paddlenlp 引入 ppmat 的版本兼容（仅路线 B 相关）。
4. WSL2 下 paddle-gpu 安装与 16GB 实测显存/步时（全部硬件数字为静态估算）。
5. 冻结 BERT 训练期 dropout 实际模式（风险 8）。
6. 训练期早停实际触发点（wandb 私有不可达）。
7. pymatgen CIF 解析绝对耗时（决定 num_workers=0 时训练是否 CPU 界）。
8. 软剪（恒 null 条件）路线的生成质量相对 unguided 训练的差异——需小规模评测。

## 13. 相对初步调研报告的修正

- mp-40 规模：127 万行 → **28,636/3,181/708 个结构**（多行 CIF 所致）。
- 「max_atoms=104」为词表大小（103 元素+1 dummy），非原子数上限；max_num_atoms=50 是死配置。
- 「radius graph/周期边界图构造依赖较重」→ 默认 edge_style="fc" 全连接，knn/PBC 整链是带 NameError 断链的死代码（~390 行可不迁）。
- 「torch_scatter 依赖」→ 零真实引用（仓库自带纯 Python scatter）。
- 「图结构、扩散过程、文本编码器依赖较重」→ 依赖面实测远轻于预期：PyG 仅 4 符号、无 C 扩展、ppmat 三路调度器均有对应物。
- 723MB ckpt ≠ 1.8 亿可训练参数；实际扩散主干 12.88M。
- constants.py 无「元素表」（原子类型即原子序数）。
