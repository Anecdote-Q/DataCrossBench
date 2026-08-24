# 评测代码与 Bench 连通性审计

审计对象：`DataCrossBench/`、`eval_with_stratification.py`、`metrics_custom.py`、`utils.py`。

## 数据链路

`DataCrossBench/flag-N/meta-info.json` 是唯一的参考答案入口，提供 `goal` 和 `insights`；`input_files` 只是参评系统的数据文件清单。原主脚本默认读取当前目录外的 `detailed_results.json`，因此原始目录不能直接用 Bench 运行。现在推荐使用：

```bash
python3 eval_with_stratification.py \
  --predictions predictions.json \
  --bench ../DataCrossBench
```

脚本会从 Bench 加载 GT，并将预测连接成内部格式。正式运行默认要求覆盖全部 200 个 flag，`--allow-subset` 仅用于调试。

## 特殊处理和风险

| 级别 | 位置 | 行为 | 影响 |
|---|---|---|---|
| 高 | `metrics_custom.py` | 四维分数依赖 Chat Completions 与 Embeddings API；输入目录中的 CSV/DB/图像不会被指标读取 | Bench 的数据内容由参评系统使用，评测器只比较文本洞察 |
| 高 | `eval_with_stratification.py` | 默认移除四种大小写敏感的洞察前缀 | 前缀本身不参与评分，其他前缀仍会参与 |
| 中 | `utils.py` | 数字匹配允许 10% 相对误差，单位必须相同；不做一对一匹配 | 年份/编号会计入；同一个预测数字可能匹配多个 GT 数字；GT 无数字时 hard 分数为 1 |
| 中 | `utils.py` | 单位正则的 `亿` 排在 `亿元` 前 | `119.66亿元` 会被解析成单位 `亿`；不同写法可能被视为不匹配 |
| 中 | `metrics_custom.py` | Completeness 是 GT 到预测的单向最大余弦相似度平均值，未裁剪 cosine | 不是对称集合相似度；异常相似度可能超出设计的 0–1 范围；Embedding 失败返回 0 |
| 中 | `metrics_custom.py` | LLM 无 `<rating>` 标签时取整段响应中的第一个 1–10 数字；找不到返回 0.5 | 解释中的数字可能被误当作评分；解析失败不是显式错误 |
| 中 | `metrics_custom.py` | rating token 的 top-logprobs 只在返回的 top-5 候选中归一化 | 加权分数依赖 API 返回的 token 形式，兼容端点可能回退到直接 rating |
| 中 | `eval_with_stratification.py` | `--resume` 的 checkpoint 位于固定 `results/checkpoint.json` | 更换模型或预测文件前必须清理旧 checkpoint |
| 低 | `utils.py` | 保留了旧的 ablation/gather 数据收集器，默认跳过 flag 15 和 37 | 这些函数不在当前 CLI 调用链中，发布时不要将其当作 Bench 入口 |
| 低 | `metrics_custom.py` | `compute_completeness` 的 `model_name` 参数未用于 Embeddings，实际固定为 `text-embedding-3-small` | 改 LLM 模型不会改变 Embeddings 模型 |

## 已做的整理

- 新增 `--predictions + --bench` 接口和多种预测 JSON 格式兼容。
- 默认拒绝不完整预测；允许显式 `--allow-subset` 调试。
- 空预测列表会进入评分，不再被静默跳过。
- 异常结果不会写入 checkpoint 的 completed 集合，后续 `--resume` 可重试。
- 修正 factuality/insightfulness 分层字段映射，汇总 CSV 与打印值不再因字段名不一致显示为 0。
- 自定义结果路径会自动创建父目录。
- 新增 `validate_benchmark.py`，检查 200 个 flag、元数据引用、分层一致性和中间目录清理状态。
