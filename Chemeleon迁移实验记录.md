# Chemeleon迁移实验记录

## 2026-10-09

### 项目初始化：仓库落地、规范消化与迁移工作量深调研

- 工作区 `/home/lkyu/baidu/Chemeleon/`：`Chemeleon/` 为 torch 原始代码（上游 main df4c674，本地核对 HEAD/工作区/fsck/远程四重一致后复制，.git 已删，只读参考），后续 paddle 镜像目录与之并列
- 规范消化：工作总结（PaddleMaterials 贡献流程与 CINN 接入调研、HIENet 迁移全记录、ppmat 注释与措辞规范）、转换对齐 prompt、CINN prompt；实验记录规范/禁 SHA/最小努力原则/疑难暂停问询已写入 memory
- 深调研（12 个并行子代理：8 维深研 + 4 路对抗验证，全程纯静态只读）：产出 `docs/Chemeleon迁移工作量调研报告.md`（13 节，全部断言带 file:line，对抗验证修正了初版参数量算术、边数公式、ckpt 作废范围等错误）
- 关键结论：
  - mp-40 真实规模 train 28,636 / val 3,181 / test 708 结构（127 万行为多行 CIF）；无归一化统计量
  - CSPNet 12.88M 参数（DiffCSP 同源，cspnet.py:1 自标），可训练合计 13.86M；723MB ckpt = 冻结 BERT 440MB + Adam + D3PM buffer 86.6MB 为主
  - composition 条件即文本（无独立编码器）；FiLM [1024,640] 是剪文本唯一失配权重层；软剪（恒 null）下 decoder 100% 可载
  - 依赖：6+1 个 torch 系第三方全部有零 torch 处置路径，零 vendor 零 C 扩展（对比 HIENet 53 文件 12,289 行 vendor）；knn/PBC 链路（~390 行）为死代码可不迁
  - 硬件：训练 B=128 峰值 5.4-6.0GB（16GB 富余）；推理 <1.6GB；采样 4000 次 decoder 前向/批（CFG）
  - 工作量量级：硬剪 ≈14-21 人日 / 完整文本版 ≈21-32 人日
- 未运行任何仓库代码；git 建仓由用户手动处理

### 模型结构讲解与全局决策（/grill-me）

- 向用户讲清：两阶段结构（CrystalClip 练文本编码器 → Chemeleon 扩散训练）、端到端输入输出、chemeleon ckpt 单文件已含全部权重（双 ckpt 仪式只是 tokenizer 载体）、耦合度三层（权重/能力/对齐链路）与「条件能力谱系」4 形态
- 十项全局决策定稿（详见 `.wayfinder/tickets/00-前置决策.md`）：文本形态 4 完整版（CLIP 一阶段不迁移、对齐期预编码解耦）/ L1+L2 / 贡献档位 B 偏 A / D3PM 模型目录自带 / diffcsp 式容器+原始 Z 词表 / CINN 首选 AST（用户规则：ppmat 无强制且 AST 可行就用 AST）/ BERT 自带最小实现 / 数据不托管 / **权重全项目不交付上游**（pdparams→Chemeleon/models/paddle，torch ckpt 在隔壁 models/torch）/ 全局最小努力（只证明复现，不跑全量）
- 用户授权删除：knn 死链、sweep/client、app 演示层、可视化等外围

### 路线图绘制（/wayfinder）

- local-markdown tracker 落 `.wayfinder/`：map.md + 5 张票据（00 前置决策已关闭 + 转换/对齐/CINN/适配贡献四张 spec 票据线性链）
- 板块实施由用户手动逐块触发（不在图内）；frontier 当前 = 转换板块 spec
