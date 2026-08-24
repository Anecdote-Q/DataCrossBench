# DataCrossBench

匿名发布快照，包含 benchmark 数据和四维评测代码。数据包含 200 个 `flag`；每个 flag 的 `meta-info.json` 提供分析目标和参考洞察，`output/` 提供参评系统需要分析的数据文件。

## 文件结构

```text
DataCrossBench-anonymous/
├── DataCrossBench/
│   ├── cleaned_insights_200.json
│   └── flag-*/
│       ├── meta-info.json
│       └── output/
└── evaluation/
    ├── eval_with_stratification.py
    ├── metrics_custom.py
    ├── utils.py
    ├── prompts/
    ├── flag_classification.json
    ├── validate_benchmark.py
    ├── AUDIT.md
    └── requirements.txt
```

`output_csv_origin/` 和 `output/processed/` 已从所有 flag 删除。发布副本不包含评测结果、checkpoint、缓存或其他实验目录。

## 安装

需要 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r evaluation/requirements.txt
export OPENAI_API_KEY="你的 API key"
export OPENAI_API_URL="https://api.openai.com/v1"  # 可选
export EVAL_MODEL="gpt-4o"                            # 可选
```

评测需要 OpenAI-compatible Chat Completions（支持 `logprobs`/`top_logprobs`）和 Embeddings API。

## 参评输出接口

参评系统输出一个 JSON 文件即可，不需要复制 GT：

```json
{
  "flag-1": {
    "pred_insights": [
      "Trend: ...",
      "Comparison: ...",
      "Extreme: ...",
      "Attribution: ..."
    ]
  },
  "flag-2": {"pred_insights": ["..."]}
}
```

也接受 `{"flag-1": ["洞察"]}`，或包含 `flag_id` 和 `pred_insights` 的记录列表。`pred_insights` 必须是字符串列表；GT 和 `goal` 会从 `DataCrossBench/flag-*/meta-info.json` 自动读取。正式评测默认要求覆盖全部 200 个 flag；调试子集时显式加 `--allow-subset`。

## 运行

```bash
python3 evaluation/eval_with_stratification.py \
  --predictions /path/to/predictions.json \
  --bench ./DataCrossBench
```

默认结果写入 `evaluation/results/`（运行时生成，不应提交）。断点恢复使用 `--resume`；更换预测文件或模型时请删除旧 checkpoint，或指定新的输出目录。

无 API 的数据检查：

```bash
python3 evaluation/validate_benchmark.py \
  ./DataCrossBench \
  --classification evaluation/flag_classification.json
```

## 评分定义与固定处理

四维权重为 Factuality 0.30、Completeness 0.25、Logic 0.20、Insightfulness 0.25，设计目标结果范围为 0–1。Factuality 是数字匹配 hard metric 与 LLM 分数各 50% 的组合；Completeness 是 GT 到预测的单向最大 embedding 相似度平均值，当前未裁剪 cosine 结果。

当前实现会移除每条洞察开头大小写敏感的 `Trend:`、`Comparison:`、`Extreme:`、`Attribution:`。数字匹配默认允许 10% 相对误差，年份/编号也会参与匹配；GT 没有数字时 hard 分数为 1。LLM 解析不到 `<rating>1-10</rating>` 时会回退到响应中的第一个 1–10 数字，找不到则返回 0.5；最终 API 失败返回 0。easy/hard 只按 `output/` 是否有图像文件分层，评测器不会读取这些数据文件来重新计算 GT。

## 托管建议

当前快照约 1.0 GB，含约 253 MB SQLite 和约 122 MB CSV，超过 GitHub 普通仓库的单文件 100 MB 限制。GitHub 需要 Git LFS 或 Release 附件；匿名 4open.science 更适合作为匿名只读快照。公开前请确认数据、图像和第三方来源的再分发许可。此目录已是可上传内容，但没有绑定任何远端或作者身份。
