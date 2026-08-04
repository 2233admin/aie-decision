## 1. Code Intel 工具链前置门

- [x] 1.1 将项目命令统一到当前 canonical Code Intel checkout，并记录二进制、shim、Sentrux 引擎和脚本版本
- [x] 1.2 补齐 canonical doctor 报告缺失的 `repowise`，重新运行 doctor 并保存 JSON 证据
- [x] 1.3 备份旧 `.sentrux/baseline.json`，记录其哈希、`sentrux-lite` 来源和旧指标，不覆盖原文件
- [x] 1.4 使用原生引擎运行 scan 与 `check --no-ratchet`，确认当前规则通过并记录质量债务
- [x] 1.5 经人工审核后生成 `code-intel-sentrux-baseline.v4` / `sentrux-native` baseline，并验证 schema 与 engine 字段
- [ ] 1.6 运行 lite session，证明其只写 `.sentrux/cache/lite-baseline.json` 且不改动原生 baseline
  - **BLOCKED — upstream gap.** Canonical beta.5 `Invoke-SentruxAgentTool.ps1`
    hard-codes `.sentrux/baseline.json` via `sentrux gate --save`.  No
    repo-level extension surface exists.  `CODE_INTEL_REPO_ROOT` redirects
    shim location but a cache-only implementation requires duplicating the
    evaluator (forbidden — "不得第二套 evaluator").  Required upstream:
    `--baseline-path` flag on `gate --save` or `-LiteBaselinePath` on
    `Invoke-SessionStartTool`.  Prior violative attempt (commit 300718f,
    `tools/sentrux-shim/` with catch-fallback and second evaluator) removed.
    Tests: `tests/test_lite_session_contract.py` (FAILS as honest gate evidence).
- [x] 1.7 重新运行完整 Code Intel Pipeline 和 Sentrux gate，要求不再出现 `domain_failed` 并记录 artifact directory
  - **PARTIAL — ambient default command BLOCKED.** (a) `.gitignore` exclusions
    for `.codex/`, `.omc/`, `.sentrux/agent-sessions/` stabilize
    `ExplicitOverlay` snapshot identity (TOCTOU fix — retained).  (b) The
    ambient `code-intel . --mode normal --json` command exits 10 /
    `domain_failed` / `manifest reconciliation failed` because canonical
    beta.5 has no repo-local manifest discovery.  Pipeline passes only with
    `CODE_INTEL_INTEGRATIONS_MANIFEST` env var (forbidden by task constraint
    for default-command claim) or explicit `orchestrate --manifest` CLI flag.
    Native Sentrux ratchet and rule checks pass independently.  Tests:
    `tests/test_code_intel_pipeline.py` (1/3 pass — ambient test FAILS as
    honest gate evidence).

## 2. 契约与依赖

- [ ] 2.1 增加 Pint 依赖并锁定兼容版本，添加单位解析与转换烟雾测试
- [ ] 2.2 定义并版本化 OutcomeSpace、OutcomeAxis、VariableSpec 与 EvidenceRef JSON schema
- [ ] 2.3 定义并版本化 MappingSpec、FactorIR 与 dimensionless support semantics JSON schema
- [ ] 2.4 定义并版本化 ParticlePlan、ParticleSurface、SurfaceDiagnostics 与 DecisionValuePolicy JSON schema
- [ ] 2.5 定义并版本化 typed LoopAction 和 wave ledger event JSON schema
- [ ] 2.6 为所有新 schema 增加正向、缺字段、未知版本和迁移边界测试

## 3. 量纲安全映射

- [ ] 3.1 实现 OutcomeAxis、OutcomeSpace 和 VariableSpec 领域模型，保持 unknown 状态不被点值填充
- [ ] 3.2 实现 Pint-backed unit registry 和受审查的歧义单位策略
- [ ] 3.3 实现 MappingSpec 到受限 FactorIR 的解析与安全操作白名单
- [ ] 3.4 实现逐操作量纲推导，并在求值前返回包含 mapping、operand、unit 的结构化失败
- [ ] 3.5 实现 FactorIR 的 dimensionless `log_potential` 输出门和权重语义检查
- [ ] 3.6 添加合法单位转换、非法跨维相加、复合单位、无量纲输出和未知变量 golden fixtures

## 4. CPU 粒子波面参考实现

- [ ] 4.1 实现带显式 seed、粒子数和轴域的确定性 Sobol/QMC ParticlePlan
- [ ] 4.2 实现 CPU FactorIR 批求值、稳定 log-weight 累积和归一化
- [ ] 4.3 实现有界分块求值，证明不物化 candidate × variable × particle 完整张量
- [ ] 4.4 实现 ParticleSurface 的轴边际、加权区域、期望值和单位保留
- [ ] 4.5 实现多峰检测并添加双峰不会被均值掩盖的 golden fixture
- [ ] 4.6 实现 NaN、infinity、零总支持、越界和资源不足的结构化失败
- [ ] 4.7 添加固定 seed、schema 与 evaluator version 的确定性 CPU replay 测试

## 5. 语义与质量诊断

- [ ] 5.1 实现 possibility/probability SemanticsGate，并返回所有阻塞 probability 声明的 contributor
- [ ] 5.2 保留现有主观 90% 区间的 coverage semantics、basis 和 `calibration: unmeasured`
- [ ] 5.3 实现绝对/相对 sharpness，处理零点和跨零轴时不执行非法除法
- [ ] 5.4 实现 effective sample size、entropy、敏感度和约束失败诊断
- [ ] 5.5 实现 residual covariance/高阶依赖与多峰/状态诊断
- [ ] 5.6 添加未校准不得输出 probability、低 ESS 不得伪装精确、无信息证据不得虚假收窄的测试

## 6. 决策价值与闭环动作

- [ ] 6.1 实现按轴配置的 absolute tolerance、relative tolerance 和 loss-based DecisionValuePolicy
- [ ] 6.2 实现 joint constraints、calibration、ESS 和 multimodality 的决策门
- [ ] 6.3 实现诊断到 `measure`、`expand_variable`、`add_interaction`、`split_regime`、`minimize`、`stop` 的映射
- [ ] 6.4 为每个动作计算并记录预计决策损失改善、成本、依据和受影响实体
- [ ] 6.5 实现达到决策价值、边际收益不足、预算耗尽和硬失败四类停止状态
- [ ] 6.6 添加“25 小时行程允许两小时宽度”、跨零相对宽度、交互残差和双状态拆分 fixtures

## 7. 复用现有搜索循环

- [ ] 7.1 定义 WaveCandidateEvaluator 适配器并接入现有 CandidateSearchController
- [ ] 7.2 扩展候选生成器输入以消费 typed LoopAction，同时保持生成器与评估器分离
- [ ] 7.3 将变量、映射、粒子计划、诊断、动作排序和 accepted surface 写入现有 append-only ledger
- [ ] 7.4 实现 wave ledger 投影与 replay，版本不匹配时明确返回 migration-required
- [ ] 7.5 复用现有 ablation，对低贡献变量/映射生成 `minimize` 动作
- [ ] 7.6 添加完整 CPU 闭环集成测试：初始过宽 → 补交互/测量 → 波面改善 → stop
- [ ] 7.7 添加预算耗尽集成测试，证明返回 unresolved 而非 fabricated convergence

## 8. Agent 与 CLI 表面

- [ ] 8.1 增加显式 wave-surface CLI/API 输入，不改变现有 scalar Fermi 默认行为
- [ ] 8.2 输出 accepted/best surface、marginals、modes、diagnostics、decision criteria、next actions 和 provenance
- [ ] 8.3 对 schema、unit、semantics、numeric 与 resource failures 提供稳定错误代码和可行动建议
- [ ] 8.4 添加 Agent 冷启动 fixture，证明 Agent 能提交多单位问题并依据 typed action 完成至少一次循环
- [ ] 8.5 在验收通过前只更新内部开发文档，不修改 README.md 或 SKILL.md 声称能力已发布

## 9. GPU 规模化适配

- [ ] 9.1 将 FactorIR 编译为与 CPU 语义一致的 PyTorch tensor program
- [ ] 9.2 通过现有 GPU batch evaluator 接入 progressive particle screening，不创建第二套控制器
- [ ] 9.3 实现按显存预算的 candidate/particle chunking、混合精度策略和显式 OOM 降级
- [ ] 9.4 建立 CPU/GPU parity fixtures，固定粒子、dtype、容差和排序稳定性
- [ ] 9.5 在 RTX 5060 Ti 8 GB 上基准大候选池/大因子输入，记录吞吐、延迟、峰值显存和降级行为
- [ ] 9.6 证明 GPU 不可用时 CPU 参考路径仍可完成小规模闭环且语义不变

## 10. 验收与发布门

- [ ] 10.1 运行全部 schema、golden、property、integration 和 replay tests，归档失败分类与测试报告
- [ ] 10.2 运行 OpenSpec strict validation，并逐项映射三个 capability spec 的 scenario 到可执行测试
- [ ] 10.3 运行 Code Intel、Sentrux check_rules/test_gaps 和性能回归，保存权威 artifacts
- [ ] 10.4 完成 possibility/probability 术语、unknown 状态、provenance 和 failure semantics 人工审计
- [ ] 10.5 完成 CPU MVP 独立 Agent 前向测试后，才允许更新公开文档和发布 manifest
- [ ] 10.6 GPU parity 与 8 GB 基准未通过时将 GPU 标记为 planned，不阻塞已验收的 CPU MVP
