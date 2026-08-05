## Why

现有费米路径能够传播标量区间，但未来事件经常同时包含时间、价格、规模等不同单位，也可能呈现偏斜、多峰或状态变化。现在需要一个先走通循环的联合波面能力，使系统能验证变量映射、保留联合不确定性，并根据诊断决定测量、补变量、增加交互或拆分状态，而不是把“区间过宽”当作唯一反馈。

## Verified Current State

- 已有标量费米区间传播、候选搜索、候选变异、消融、账本和确定性回放，可作为控制面与参考路径复用。
- 已有 GPU 候选批评估变更正在实施，本变更不重复建设其调度和显存管理。
- 当前没有多轴结果空间、量纲安全映射、粒子波面或由波面诊断驱动的闭环动作；这些均为计划能力，不是现状声明。
- Code Intel 的 `domain_failed` 已确认为旧 `sentrux-lite` baseline 与当前原生 baseline 契约不兼容，并伴随旧 beta.1 脚本与 beta.5 二进制混用；原生扫描和无 ratchet 规则检查可通过。

## What Changes

- 新增带单位和时间语义的多轴 `OutcomeSpace`、`VariableSpec` 与可编译 `MappingSpec`。
- 新增加权粒子联合波面，支持边际、峰值、熵、有效样本量、敏感度、残差和多峰诊断。
- 新增决策价值策略，同时考虑轴相关的绝对宽度、相对宽度和损失函数。
- 新增诊断到闭环动作的可执行映射：补测量、补变量、增加交互/潜变量、拆分状态、最小化或停止。
- 明确区分未校准的 `possibility_surface` 与具备声明且已验证校准依据的 `probability_surface`。
- 先交付 CPU 确定性参考闭环，再通过已有 GPU 批评估接口扩展规模；不物化完整候选 × 变量 × 粒子张量。
- 将 Code Intel/Sentrux 版本与 baseline 对齐作为实施前置门，不静默覆盖旧 baseline。

## Non-goals

- 不自动宣称发现因果关系，不保证精确单点或日期。
- 不实现完整物理方程或连续 PDE 求解器。
- 不让 LLM 输出直接成为计算事实；生成器只提出候选，评估器和证据契约负责裁决。
- 不把主观或未校准输入产生的相对权重称为经验概率。

## Capabilities

### New Capabilities

- `dimension-aware-factor-mapping`: 定义多轴结果、带单位变量、因子映射、量纲验证与无量纲势函数编译。
- `joint-wave-surface`: 定义带权粒子波面、可能性/概率语义、摘要、诊断和可追溯输出。
- `wave-surface-search-loop`: 定义由决策价值与诊断驱动的循环动作、停止条件、账本和确定性回放。

### Modified Capabilities

- 无。本变更通过适配器复用进行中的候选搜索和 GPU 批评估能力，不修改其既有需求。

## Authority

- 产品行为以本变更的规范和 `docs/PRD-JOINT-WAVE-SURFACE.md` 为准。
- CPU 参考求值器是数值正确性的首个验收权威；GPU 结果必须在声明容差内与其一致。
- 证据、变量、映射和波面语义以版本化 schema 为准，生成器输出不具有事实权威。

## Dependencies

- 复用现有 `search.py`、`candidate_generation.py`、`fermi.py`、账本和回放能力。
- 使用 Pint 处理单位与量纲；使用成熟的 Sobol/QMC 与张量库完成采样和批求值。
- 规模化依赖 `add-gpu-candidate-search-loop` 的稳定批评估接口，但 CPU MVP 不被其阻塞。
- 实施前需将 Code Intel/Sentrux 统一到当前 canonical 工具链，补齐 `repowise`，并经人工审核迁移原生 v2 baseline。

## Impact

- 新增领域模型、schema、CPU 求值器、诊断器、循环策略和 CLI/Agent 输入输出。
- 为现有搜索控制器增加波面评估适配器，不移除标量费米路径。
- 增加 Pint，并复用项目已有或经批准的 NumPy/SciPy/PyTorch 依赖。
- 增加多单位、多峰、交互残差、回放、语义门和 CPU/GPU 一致性 fixtures。
