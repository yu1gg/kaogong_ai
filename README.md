# AI 公考面试测评项目

## 1. 项目定位

这是一个面向“公考面试作答测评”的后端项目，目标不是简单让大模型直接打分，而是把评分拆成更可控的工程链路：

1. 接收文本、音频、视频作答。
2. 将音视频转成可分析的 `transcript`。
3. 根据 `question_id` 读取题库配置。
4. 先做“证据抽取”，再做“证据约束评分”。
5. 对模型输出做确定性校验、修正和收敛。
6. 支持回归测试、LLM 标定和题库样本迭代。

当前核心代码位于 [ai_gongwu_backend](/home/quyu/ai_interview/ai_gongwu_backend)。

---

## 2. 当前状态

截至 2026-04-11，项目已经完成这些核心能力：

1. FastAPI 接口可提供文本、音频、视频测评。
2. 题库支持目录式多题加载，不再局限单题。
3. 评分链路已经升级为“两阶段证据评分”：
   - 第一阶段：只抽取证据
   - 第二阶段：只基于证据打分
4. 抽象扣分 / 加分理由必须绑定 `evidence_ids`。
5. 后处理会校验：
   - 维度名是否合法
   - 分项分与总分是否一致
   - 引用证据是否能在原文中核验
   - 缺失型扣分是否真的有对应 absence 证据
6. 已支持测评结果落库和记录查询。
7. 已新增湖南题库自动导入链路，并生成高 / 中 / 低三档回归样本。
8. 已提供：
   - 确定性回归脚本
   - 真实 LLM 回归 / 标定脚本
   - `repeat` 多次采样取中位数能力
   - `writeback` 回写 `llmExpectedMin/Max` 能力

当前仓库内除了手工题库外，还包含一套自动生成的湖南题库：

1. 题库目录：`ai_gongwu_backend/assets/questions/generated_hunan/`
2. 样本目录：`ai_gongwu_backend/assets/regression_samples/generated_hunan/`
3. 导入脚本：`ai_gongwu_backend/scripts/import_hunan_question_bank.py`
4. 当前已生成 `29` 道题，每题配套 `high / mid / low` 三档回归样本。
5. 导入摘要见：`ai_gongwu_backend/assets/questions/generated_hunan/import_summary.txt`

---

## 3. 目录结构

建议先从下面这个结构理解项目：

```text
ai_interview/
├── README.md
├── docs/
│   ├── 项目修改记录.md
│   └── 回归与标定说明.md
├── sample_sets/
│   └── README.md
├── reports/
│   └── regression/
│       ├── README.md
│       └── *.json / *.md
├── ai_gongwu_backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints/
│   │   │       └── interview.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── dependencies.py
│   │   ├── models/
│   │   │   └── schemas.py
│   │   ├── services/
│   │   │   ├── flow.py
│   │   │   ├── evaluation_store.py
│   │   │   ├── question_bank.py
│   │   │   ├── llm/
│   │   │   │   └── client.py
│   │   │   ├── media/
│   │   │   └── scoring/
│   │   │       ├── prompts.py
│   │   │       └── calculator.py
│   │   └── main.py
│   ├── assets/
│   │   ├── questions/
│   │   │   ├── README.md
│   │   │   ├── HN-LX-20200606-01.json
│   │   │   └── generated_hunan/
│   │   └── regression_samples/
│   │       ├── README.md
│   │       └── generated_hunan/
│   ├── scripts/
│   │   ├── import_hunan_question_bank.py
│   │   ├── run_regression.py
│   │   └── run_llm_regression.py
│   ├── tests/
│   ├── requirements.txt
│   └── run.sh
└── 湖南题库源文档 / extracted 文本 / 若干手工样本
```

---

## 4. 三条核心链路

### 4.1 测评链路

核心入口是 [flow.py](/home/quyu/ai_interview/ai_gongwu_backend/app/services/flow.py)。

完整流程如下：

1. 接口层接收 `question_id` 和输入文件。
2. 文本直接进入评分；音视频先经过媒体解析。
3. 根据 `question_id` 从题库中读取单题配置。
4. 第一阶段 Prompt 只抽取原文证据。
5. 系统对证据做对齐、补充 absence 证据、结构校验。
6. 第二阶段 Prompt 只基于证据包打分。
7. `calculator.py` 做确定性后处理。
8. 若允许持久化，则把 Prompt、原始输出、最终结果一起落库。

### 4.2 题库导入链路

核心脚本是 [import_hunan_question_bank.py](/home/quyu/ai_interview/ai_gongwu_backend/scripts/import_hunan_question_bank.py)。

作用：

1. 读取仓库根目录下的湖南题库源文档提取文本。
2. 解析题干、评分标准、扣分标准、参考答案、标签等信息。
3. 自动生成题目 JSON。
4. 自动生成高 / 中 / 低三档回归样本。
5. 为每题写入：
   - `scoreBands`
   - `regressionCases`
   - `llmExpectedMin`
   - `llmExpectedMax`

### 4.3 回归 / 标定链路

当前有两套脚本：

1. [run_regression.py](/home/quyu/ai_interview/ai_gongwu_backend/scripts/run_regression.py)
   用于确定性 / 常规链路回归。
2. [run_llm_regression.py](/home/quyu/ai_interview/ai_gongwu_backend/scripts/run_llm_regression.py)
   用于真实大模型回归、区间标定和回写。

推荐顺序：

1. 先重导入题库和样本。
2. 先跑 `run_regression.py` 看生成样本排序是否合理。
3. 再跑 `run_llm_regression.py --repeat 3` 看真实模型中位数。
4. 结果稳定后，再加 `--writeback` 回写 `llmExpectedMin/Max`。

---

## 5. 关键模块说明

### 5.1 [interview.py](/home/quyu/ai_interview/ai_gongwu_backend/app/api/endpoints/interview.py)

接口层，负责：

1. 题目列表与题目详情接口
2. 文本测评接口
3. 音视频测评接口
4. 测评记录列表与详情接口

### 5.2 [schemas.py](/home/quyu/ai_interview/ai_gongwu_backend/app/models/schemas.py)

数据合同层，定义：

1. 题目结构
2. 两阶段证据结构
3. 评分结果结构
4. 题目分档与回归样本结构

当前题库回归相关字段主要是：

1. `scoreBands`
2. `regressionCases`
3. `llmExpectedMin`
4. `llmExpectedMax`

### 5.3 [prompts.py](/home/quyu/ai_interview/ai_gongwu_backend/app/services/scoring/prompts.py)

Prompt 构造层，负责：

1. 第一阶段证据抽取 Prompt
2. 第二阶段证据约束评分 Prompt
3. 按题目动态生成本土化 / 岗位化提示，不再写死河南模板

### 5.4 [calculator.py](/home/quyu/ai_interview/ai_gongwu_backend/app/services/scoring/calculator.py)

确定性后处理层，负责：

1. 证据对齐
2. absence 证据补充
3. 理由与证据绑定校验
4. 分数收敛和排序校准
5. 规则型兜底评分

---

## 6. 环境准备

推荐使用 Python 3.10 及以上版本。

```bash
cd /home/quyu/ai_interview
python3 -m venv .venv
source .venv/bin/activate
pip install -r ai_gongwu_backend/requirements.txt
```

如果你已经在后端目录里维护虚拟环境，也可以直接使用：

```bash
cd /home/quyu/ai_interview/ai_gongwu_backend
./venv/bin/pip install -r requirements.txt
```

系统依赖建议：

1. `ffmpeg`
2. `torch`
3. `openai-whisper`
4. `opencv-python-headless`

---

## 7. 环境变量

建议在 [ai_gongwu_backend](/home/quyu/ai_interview/ai_gongwu_backend) 下准备 `.env`：

```env
LLM_PROVIDER=QWEN
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen3-coder-plus

QUESTION_DB_PATH=assets/questions
WHISPER_MODEL_SIZE=base
WHISPER_CPU_THREADS=4
WHISPER_LANGUAGE=zh
ENABLE_VISUAL_ANALYSIS=true

MIN_VALID_WORDS=15
MIN_WORDS_PENALTY_RATIO=0.2
SCORE_TOLERANCE=2.0
MAX_RATIONALE_CHARS=400
```

说明：

1. `QUESTION_DB_PATH` 现在默认应指向目录，而不是单个题目文件。
2. 若未配置 `LLM_API_KEY`，系统会回退到确定性评分兜底。

---

## 8. 启动项目

### 方式一：标准命令启动

```bash
cd /home/quyu/ai_interview/ai_gongwu_backend
uvicorn app.main:app --reload --port 9000
```

### 方式二：直接运行入口文件

```bash
python /absolute/path/to/ai_gongwu_backend/app/main.py
```

也支持在 `ai_gongwu_backend` 目录内使用模块方式启动：

```bash
python -m app.main
```

### 方式三：运行脚本

```bash
cd /home/quyu/ai_interview/ai_gongwu_backend
bash run.sh
```

启动后访问：

```text
http://127.0.0.1:9000/docs
```

---

## 9. 主要接口

### 9.1 题目接口

1. `GET /api/v1/interview/questions`
2. `GET /api/v1/interview/questions/{question_id}`

### 9.2 测评接口

1. `POST /api/v1/interview/evaluate`
   音频 / 视频输入
2. `POST /api/v1/interview/evaluate/text`
   文本输入

### 9.3 测评记录接口

1. `GET /api/v1/interview/records`
2. `GET /api/v1/interview/records/{record_id}`

---

## 10. 常用脚本

### 10.1 重新导入湖南题库

```bash
cd /home/quyu/ai_interview/ai_gongwu_backend
./venv/bin/python scripts/import_hunan_question_bank.py
```

执行后会刷新：

1. `assets/questions/generated_hunan/`
2. `assets/regression_samples/generated_hunan/`
3. `assets/questions/generated_hunan/import_summary.txt`

### 10.2 跑确定性回归

```bash
cd /home/quyu/ai_interview/ai_gongwu_backend
./venv/bin/python scripts/run_regression.py
```

### 10.3 跑真实 LLM 回归

```bash
cd /home/quyu/ai_interview/ai_gongwu_backend
./venv/bin/python scripts/run_llm_regression.py --repeat 3
```

### 10.4 跑真实 LLM 回归并回写区间

```bash
cd /home/quyu/ai_interview/ai_gongwu_backend
./venv/bin/python scripts/run_llm_regression.py --repeat 3 --writeback
```

说明：

1. `--repeat 3` 用多次采样中位数抵消单次模型波动。
2. `--writeback` 会把新的 `llmExpectedMin/Max` 回写到题目 JSON。
3. 真正批量标定前，建议先用 `--question-id` 跑小子集。

---

## 11. 题库与样本文件怎么理解

### 11.1 单题 JSON

每题一个 JSON，至少包含：

1. `id`
2. `type`
3. `province`
4. `fullScore`
5. `question`
6. `dimensions`

建议同时包含：

1. `scoreBands`
2. `regressionCases`
3. `sourceDocument`
4. `referenceAnswer`
5. `tags`

### 11.2 regressionCases

当前约定每题至少有 3 条：

1. 文档高分基准答案
2. 程序化中档参考答案
3. 程序化低档参考答案

其中：

1. `expected_min/max` 主要对应当前确定性排序结果
2. `llmExpectedMin/Max` 对应真实大模型回归后的推荐区间

### 11.3 generated_hunan

这是脚本自动生成目录，不建议手工逐个改动。  
如果你调整了导入或样本生成规则，应重新运行导入脚本，让生成产物整体刷新。

---

## 12. 当前文档索引

1. [docs/项目修改记录.md](/home/quyu/ai_interview/docs/项目修改记录.md)
2. [docs/回归与标定说明.md](/home/quyu/ai_interview/docs/回归与标定说明.md)
3. [ai_gongwu_backend/assets/questions/README.md](/home/quyu/ai_interview/ai_gongwu_backend/assets/questions/README.md)
4. [ai_gongwu_backend/assets/regression_samples/README.md](/home/quyu/ai_interview/ai_gongwu_backend/assets/regression_samples/README.md)
5. [reports/regression/README.md](/home/quyu/ai_interview/reports/regression/README.md)
6. [sample_sets/README.md](/home/quyu/ai_interview/sample_sets/README.md)

---

## 13. 当前注意事项

1. `sample_sets/` 主要是早期人工整理的河南样本分类目录，不等同于当前 `generated_hunan` 回归集。
2. 真实 LLM 分数仍然会抖动，所以正式标定建议固定使用 `--repeat 3`。
3. 组织策划题和综合分析题现在已经拆成独立的中低档模板生成器，但仍建议持续做题型级微调。
4. 如果修改了：
   - `prompts.py`
   - `calculator.py`
   - `import_hunan_question_bank.py`
   最好同步重跑导入和回归，而不是只测单条接口。

---

## 14. 后续建议

当前最值得继续推进的方向：

1. 对剩余题型继续做模板化样本生成，而不是统一抽句降档。
2. 把全量 `repeat=3` 标定跑完，并持续回写 `llmExpectedMin/Max`。
3. 为关键题型建立人工标注对照集，校验 LLM 中位数与人工目标分差。
4. 给回归脚本补题型分组统计和异常样本自动汇总。
5. 对测评记录增加“人工复核状态”和“版本号”字段，便于后续追踪。
