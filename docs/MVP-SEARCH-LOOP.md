# Fermi Search Loop MVP

这个实验入口先验证完整线路，不代表候选概率或区间已经校准。

## 运行

```powershell
uv run aie-decision search-fermi fixtures/search/v1/mvp-loop.json --experimental
```

输入包括：

- `variables`：已有变量区间与测量方法；只有显式设置 `ablatable: true` 的变量允许自动消融。
- `candidates`：初始候选公式和先验权重。
- `mutation_templates`：失败诊断触发的公式模板；以后可由 LLM proposer 替换，但仍经过本地校验、去重和标记。
- `budget`：候选数、轮数、评估次数与墙钟时间硬上限。

## 当前闭环

```text
SEED -> VALIDATE -> EVALUATE -> RANK
  ^                                  |
  +---- EXPAND/REVISE <- failed -----+
                         passed -----+-> MINIMIZE -> RESULT/STOP
```

候选排序使用未校准的 Bayes-inspired 分数：

```text
pseudo_posterior = prior_weight * heuristic_likelihood / sum(weights)
```

它只用于搜索排序，不是统计学后验概率。输出固定包含 `calibrated: false`、不可变事件账本、可验证 checkpoint、候选谱系、区间和停止原因。

## MVP 边界

- 候选自动生成目前使用声明模板；真实 LLM 仅预留 `CandidateProposer` 接口。
- 数值计算使用 CPU 区间算术；没有 GPU 批处理。
- 找到闭环结果时输出 `experimental_usable: true`，可用于 MVP 试用；正式 `usable` 仍保持 `false` 和 `provisional_uncalibrated`，直到校准验收完成。
- 原始数据抓取、外部行动和 HMC 集成都不在此入口内。

## 后续执行架构

执行器保留两条路径：小候选池继续走当前 CPU 参考执行器；大候选池以后接入 GPU batch evaluator，但不改变搜索状态机。GPU 路径只装载当前批次实际引用的因子和样本，禁止构造“全部候选 × 全部因子 × 全部样本”的稠密张量。

失败不会被包装成成功结果：候选无效记为 `REJECT`；没有候选达标返回 `insufficient-information`；达到硬预算返回 `budget-exhausted`；完成最小化才返回 `result-found`。GPU 或外部 proposer 不可用时，后续适配器必须明确拒绝或降级，不能静默更换语义。
