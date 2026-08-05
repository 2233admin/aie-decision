# AIE Decision

AIE Decision v1 是一个供 AI 调用的、可执行且可回放的费米分解运行时。用户只需要给原始定量问题；AI 通过公开动作把目标递归拆成可观察、可计数或可测量的叶子，运行时负责量纲、概率传播、最小前沿实验、预算、回滚和审计。

它不是公式填写器、固定行业估算器或 prompt 角色扮演。仓库中原有的固定吞吐量产品切片已经移除；`fermi.py` 只作为旧的确定性区间算术基础设施保留，不是产品入口。

## 已实现的纵切

```text
原始问题
  → AI 定义目标与可接受宽度
  → 递归分解 / 替代分支
  → 结构化原子测量与叶子边际
  → 显式联合假设与 Monte Carlo 传播
  → 目标 P05 / P50 / P95 和区间宽度
  → 逐叶删除必要性验证
  → 最高价值测量的条件饱和性验证
  → certified / insufficient 证书
  → 追加式轨迹、回滚和确定性 replay
```

概率依据不足时，叶子必须使用 `unknown` marginal；这时系统只输出明确标记的场景边界，不输出或认证目标 90% 概率区间。联合依赖未声明时同样不能认证。

## 命令行

安装后，默认命令就是集成后的 Fermi 运行时：

```powershell
aie-decision discover

aie-decision start `
  --session-id demo `
  --question "一个典型工作日，某城市地铁乘客支付多少票款？" `
  --session .\session.json
```

`discover` 返回版本化动作合同和当前合法动作。AI 随后用 JSON 文件执行动作：

```powershell
aie-decision apply --session .\session.json --input .\action.json
aie-decision inspect --session .\session.json
aie-decision finalize --session .\session.json
aie-decision replay --session .\session.json
```

`action.json` 同时支持扁平形式和协议 envelope：

```json
{"action":"evaluate"}
```

```json
{"action":"evaluate","payload":{}}
```

完整、可执行的版本化动作轨迹见 [subway-fares.json](examples/fermi-runtime-v1/subway-fares.json)。Python 调用入口是 `aie_decision.FermiKernel` 与 `aie_decision.agent_runtime.AgentRuntime`。

## 真实 AI 验收

一次隔离的 Claude Code 冷启动只收到陌生原始问题，不能读取仓库文件，只能通过 `aie-decision discover` 学习接口。它自行完成两层递归分解、四个原子叶子、联合假设、区间传播、删除实验、饱和性和 replay。目标区间虽然满足其声明宽度，但最高价值测量仍可显著缩窄区间，因此运行时拒绝认证并返回 `insufficient`。

这项验收证明的是“新 AI 能否使用工具并接受运行时约束”，不证明 AI 提供的外部事实准确，也不证明经验覆盖率已经校准。详见 [REAL-AI-ACCEPTANCE.md](docs/REAL-AI-ACCEPTANCE.md)。

## v1 的诚实边界

- AI 负责理解问题、提出分解和提供证据/假设；运行时不内置模型供应商或研究 prompt。
- v1 数值评估器支持受限代数关系、常数、正态/对数正态分位拟合和显式未知边际。
- v1 联合模型支持独立或全局正/负等相关假设；尚不支持任意相关矩阵或条件依赖图。
- 必要性执行逐叶删除后的可回答性验证；饱和性执行当前最高价值叶子的精确测量反事实。证书明确是条件性的，不声称找到全局最优或绝对最小分解。
- 来源、测量程序和区间 rationale 会被保存和审计，但 v1 不负责联网核验来源。
- 无校准数据时，结果标记为 `unmeasured`，不得解读为已验证的经验 90% 覆盖率。

产品方向见 [FERMI-PRODUCT-DIRECTION.md](docs/FERMI-PRODUCT-DIRECTION.md)，实现规格见 [build-recursive-fermi-runtime](openspec/changes/build-recursive-fermi-runtime)。

## 验证

```powershell
uvx --from pytest pytest -q
openspec validate build-recursive-fermi-runtime --strict
```

## License

MIT
