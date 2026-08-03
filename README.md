# AIE Decision

AIE Decision v1 是一个可运行、可审计的费米估算纵切版本。它只支持一个明确领域：**运营吞吐量**。用户输入原始问题和带来源标识的材料，系统负责抽取量、生成并比较分解、验证最小变量集、传播不确定性，并指出下一项最值得测量的变量。

它不会要求用户先写公式、变量表或上下界，也不会在材料不足时编造证据。

## 运行

需要 Python 3.12 与 [uv](https://docs.astral.sh/uv/)：

```powershell
uv run aie-decision estimate fixtures/throughput/v1/cafe-day.json
```

输入只有问题和材料；`coverage`、`samples`、`seed` 是可选运行控制：

```json
{
  "question": "How many orders can North Cafe process per day?",
  "materials": [
    {
      "id": "operations-note",
      "text": "North Cafe has 2 service stations. Each station can process 18 orders per hour. The cafe operates 8 hours per day. Utilization has a stated 90% probability interval of 70% to 90%."
    },
    {
      "id": "demand-note",
      "text": "Daily demand has a stated 90% probability interval of 180 to 260 orders per day."
    }
  ]
}
```

## 输出契约

完整结果包含：

- 从原文抽取的叶子变量、句子定位、证据类型与概率分布；
- 理论产能、有效产能、需求量和供需瓶颈等候选分解及完整性比较；
- 对选中分解逐个删除叶子的实际可回答性测试；
- 显式联合假设下 Monte Carlo 得到的目标 P05/P50/P95 和 90% 区间宽度；
- 正相关与反向秩相关压力情景；
- 逐变量“测准后”反事实重算得到的预期缩窄量和下一测量项；
- 未建立的校准、依赖假设和材料缺口。

普通数值范围不会自动被当成 90% 概率区间。材料没有给出所需量或概率语义时，命令以 `partial` / `not_answerable` 结束，并列出缺口。

## 当前边界

- 仅支持英文运营吞吐量材料的受控语法，不宣称通用文本理解。
- 只使用用户提供的材料，不做外部数据采集。
- 主结果的独立性是显式但未验证的联合假设；压力情景用于暴露依赖敏感性。
- 没有历史结果时，`calibration` 固定为 `unmeasured`。
- `fermi` 子命令保留为手工公式的确定性区间算术基础设施，不是 v1 产品验收路径。

## 验证

```powershell
uv run --with pytest pytest -q
```

## License

MIT
