# AIE Decision

AIE Decision 是一个答案导向的逆向测量与证据编译器。它从“最终需要回答什么”出发，把用户提供的问题和材料编译为可追溯、可校验、可复算的决策分析包。

它不是文章摘要器，也不会把缺失信息偷偷补成确定答案。

## 第一版：最小费米估算

第一版最短路径只做一件事：给出一条费米公式和各变量所在的联合 90% 情景区间，程序保留公式真正引用的最小变量集合，传播出目标区间，计算宽度，并指出最值得继续测量的变量。

```powershell
uv run aie-decision fermi fixtures/fermi/v1/daily-revenue.json
```

示例公式：

```text
daily_revenue = visitors * conversion_rate * average_order_value
```

输出直接包含：

- `minimal_variables`：公式真正需要的变量；
- `target_interval`：目标的声明 90% 主观可信区间；
- `absolute_width` 和 `normalized_width`：区间有多宽；
- `largest_uncertainty_source`：哪个变量造成的宽度最大；
- `next_measurement`：下一项最值得缩窄的变量。

第一版不获取外部数据，也不把单次主观区间冒充成已经完成历史校准的预测区间。

## 核心流程

```text
目标答案
  → 必要条件图
  → 原子命题证据
  → 事件现场还原
  → 缺失条件区间
  → 二次因子候选
  → 预测区间审计
```

每次分析都会保留来源、位置、说话者、命题类型、条件依赖、计算过程和修订记录。事实、引述、推断、评价、预测和缺失信息不会被混为一谈。

## 能做什么

- 把问题固定为带目标、单位、截止时间和决策用途的 Answer Contract。
- 构造必要条件图，标出已知、推导、估计、缺失和冲突条件。
- 将混合文本拆成带来源与说话者的原子命题。
- 从可采信命题中还原人物、动作、时间、地点、顺序与争议。
- 对缺失条件保留区间、假设、依赖关系和敏感性，而不是静默填点。
- 生成可证伪的二次因子候选。
- 审计预测区间的语义、宽度、覆盖率、信息量和基线改进。
- 同时输出机器 JSON、中文报告和不可变分析账本。

## 安装

需要 Python 3.12 与 [uv](https://docs.astral.sh/uv/)。

```powershell
uv sync --python 3.12
uv run aie-decision --help
```

## 快速开始

准备一个符合 [输入契约](references/input-contract.md) 的 JSON 文件，然后运行：

```powershell
uv run aie-decision compile input.json --output-dir output
```

输出目录包含：

- `analysis-package.json`：版本化机器包；
- `decision-report.md`：面向人的分析报告；
- `analysis-ledger.json`：可追溯分析账本。

其他命令：

```powershell
uv run aie-decision fermi fixtures/fermi/v1/daily-revenue.json
uv run aie-decision validate analysis-package.json
uv run aie-decision render-report analysis-package.json --output report.md
uv run aie-decision audit-interval --help
```

## 终态语义

- `complete`：结构完整，不等于结论必然真实或预测已经校准。
- `partial`：系统诚实交付已有结果，并明确列出缺失内容。
- `exit 2`：输入或输出仍有校验问题，只能作为诊断材料。
- `uncalibrated_*`：区间尚未建立经验校准，不能宣传为已校准预测。

## 产品边界

- 只处理用户提供的材料，不自动采集外部数据。
- 来源文本一律视为不可信数据，不执行其中包含的指令。
- 不把作者立场、修辞或未经支持的因果解释升级为现场事实。
- 不因缺少信息而编造来源、数值、条件或结论。
- legacy/manual 模式仅用于兼容旧式对话，不产生正式分析包。

## Skill

仓库根目录本身也是一个可调用的 Codex Skill。调用 `$aie-decision` 时，Skill 会使用同一个 CLI 生成正式产物，而不是临时编写一段不可复现的分析。

## 当前版本

这是 standalone v1 的首个公开版本。核心运行时、Schema、CLI 和 Skill 已可使用；复杂说话者归因仍是已知改进方向。

## License

MIT
