# 回归报告目录

这里用于存放两类批量回归脚本输出的报表。

## 1. 确定性回归

脚本：

```bash
cd /home/quyu/ai_interview/ai_gongwu_backend
./venv/bin/python scripts/run_regression.py
```

输出文件：

1. `regression_时间戳.json`
2. `regression_时间戳.md`

适用场景：

1. 刚导入完题库，先检查高 / 中 / 低三档样本排序是否合理
2. 调整 `calculator.py`、样本生成逻辑后，先做快速体检
3. 在没有可用 LLM 密钥时，先做链路级回归

常用参数：

1. `--question-id`
   只跑指定题目，可重复传入
2. `--persist`
   是否把回归结果也落库

## 2. 真实 LLM 回归

脚本：

```bash
cd /home/quyu/ai_interview/ai_gongwu_backend
./venv/bin/python scripts/run_llm_regression.py --repeat 3
```

输出文件：

1. `llm_regression_时间戳.json`
2. `llm_regression_时间戳.md`

适用场景：

1. 需要验证真实大模型在当前 Prompt 和规则下的稳定性
2. 需要根据实测结果标定 `llmExpectedMin / llmExpectedMax`
3. 需要观察多次采样波动，而不是只看单次结果

常用参数：

1. `--question-id`
   只跑指定题目，可重复传入
2. `--sample-level {all,high,mid,low}`
   只跑某个档位样本
3. `--repeat 3`
   每个样本重复 3 次，取中位数作为实测分
4. `--allow-fallback`
   允许 LLM 失败后回退到确定性评分
5. `--writeback`
   把当前实测结果生成的新 `llmExpectedMin / llmExpectedMax` 回写题库 JSON
6. `--persist`
   是否把回归结果也落库

## 3. 推荐执行顺序

推荐按下面顺序执行，而不是直接全量跑真实 LLM：

1. 先执行 `scripts/import_hunan_question_bank.py`
2. 再执行 `scripts/run_regression.py`
3. 子集通过后，执行 `scripts/run_llm_regression.py --repeat 3`
4. 小范围题目稳定后，再执行 `scripts/run_llm_regression.py --repeat 3 --writeback`

## 4. 如何看 JSON 报表

LLM 回归和确定性回归都包含 `summary` 与 `rows`。

重点字段建议优先看：

1. `summary.pass / fail / error / skip`
   先看总体是否已经可接受
2. `expected_range`
   当前样本的目标区间
3. `expectation_source`
   LLM 回归里表示当前区间来自：
   - `llmExpected`
   - `deterministicExpected`
4. `attempt_scores`
   只有 LLM 回归在 `repeat > 1` 时会出现，表示每次实测分数
5. `fallback_used`
   是否发生了确定性回退
6. `validation_issue_count`
   结果经过多少条后处理校验提示

## 5. 当前建议

1. 真正做标定时，默认使用 `--repeat 3`
2. 回写前先用 `--question-id` 只跑子集
3. 如果出现大面积 `ERROR`，先检查 `LLM_API_KEY`、模型配置和网络，而不是直接相信报表结论
