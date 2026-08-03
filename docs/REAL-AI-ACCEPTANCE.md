# Real-AI Cold-Start Acceptance

日期：2026-08-04
验收对象：`build-recursive-fermi-runtime`

## 隔离条件

- 新建空工作目录，不在 Git 仓库中运行。
- Claude Code 不能读取仓库源码、测试、README 或示例。
- PATH 中只提供已安装的 `aie-decision` 命令。
- 初始业务输入只有问题：“2026 年一个典型工作日，上海地铁站内的自动售货机一共完成多少笔饮料购买交易？”
- 未提供公式、变量、叶子、上下界或候选分解。

Claude 只能通过 `aie-decision --help` 和 `aie-decision discover` 学习协议，然后自行创建 JSON 动作和持久化 session。

## 观察到的轨迹

最终 session 包含 38 个追加式事件。AI 完成：

- `define_question`；
- 三次有效 `expand`，形成深度 2 的递归树；
- 四次有效 `propose_atom`；
- `set_dependence`、`evaluate`、`test_frontier`、`finalize`；
- `replay`。

另有 6 个被拒绝的动作，原因包括未注册单位、量纲不闭合和 derived proxy 缺少 `assumption_notes`。AI 读取结构化错误后修正动作，没有修改运行时或绕过验证。

有效分解为：

```text
工作日饮料交易
  = 地铁站内售货机数量 × 单机工作日交易
  = 运营车站数 × 每站机器数 × 单机全日交易 × 工作日系数
```

四个叶子均记录了对象、范围、单位、来源、测量程序、分位区间和假设。它们属于未校准的估计/代理，不是经过本验收独立核实的事实。

## 数值与前沿结果

- 声明的联合模型：全局正相关，相关系数 `0.3`，5,000 个样本，seed `42`。
- 目标 P05 / P50 / P95：约 `7,150 / 22,142 / 69,051` 笔。
- 90% 区间宽度：约 `61,901` 笔。
- 概率语义：`monte_carlo_joint_sampling`。
- 校准标签：`unmeasured`。
- 删除验证：删除任一保留叶子都会令当前关系不可计算。
- 下一项最值得测量：单机全日饮料交易数，预计缩窄约 `39,257` 笔，占当前宽度约 `63.4%`。
- 饱和性：`false`。
- 前沿证书：`structurally_complete`，但 `certified=false`。
- finalize：`insufficient`。
- replay：`match`，无 mismatch。

这证明运行时能让一个新 AI 从原始问题进入真实工具循环，并能在“区间已经够宽但继续测量仍很有价值”时拒绝假完成。

## 验收推动的修复

更早一次冷启动暴露并推动了两项接口修复：

1. `apply` 同时接受扁平动作 JSON 和 `payload` envelope；
2. `define_question` 接受后立即创建并暴露可寻址根节点 `n_0001`。

最终验收后又收紧了联合假设合同：`set_dependence` 必须提供 rationale，并明确 v1 的单个相关系数是对所有非恒定叶子生效的全局等相关假设。

## 不构成的证明

- 不证明 Claude 提供的外部来源或数字真实；
- 不证明目标区间具有已校准的经验 90% 覆盖率；
- 不证明当前分解是所有可能分解中的全局最优；
- 不证明 v1 已支持任意相关矩阵、条件依赖或自动外部检索。
