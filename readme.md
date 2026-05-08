# Harness Engineering 考核探索报告
 
> 探索时间：2026年5月  
> 模型：Qwen3-8B (阿里云百炼)  
> 配置：workers=10, max_prompt_tokens=2048

---

## 一、任务背景与目标

本次考核的任务是设计一个Harness，使LLM在有限输入窗口（2048 tokens）的文本分类任务上达到尽可能高的准确率。

### 1.1 考核要求

- 设计一个含有外部记忆管理、预算控制的Harness
- 通过`update(text, label)`接收训练样本
- 通过`predict(text)`进行预测
- 模型权重不变，所有"学习"发生在Harness维护的外部状态中

### 1.2 数据集说明

**官方DEV集**：创智官方提供的银行客服意图分类数据集，231条训练样本，77类标签，每类约3条。

**自构建验证集**：为验证Harness的泛化能力，我自行构建了三个额外的验证集：


| 任务类型    | 构建来源           | 数据规模         | 特点            |
| ------- | -------------- | ------------ | ------------- |
| OOD集    | 其他领域intent分类任务 | 训练42/测试113条  | 完全不同的领域和标签体系  |
| Other集  | 收集的intent分类样本  | 训练300/测试500条 | 多样化intent混合领域 |
| Choice集 | 自然语言问答选择题      | 100+道选择题     | A/B/C/D选项格式   |


这些数据集用于：

1. 验证任务类型检测逻辑是否正确
2. 测试不同prompt策略在不同领域的效果
3. 迭代优化时快速验证修改是否有效

### 1.3 评测指标

- DEV: 84.6% (目标)
- OOD: 90.3% (目标)
- Choice: 97.3% (目标)

---

## 二、完整版本历史


| 版本      | 策略                         | DEV准确率     | 其他数据集                                   | 主要改动            |
| ------- | -------------------------- | ---------- | --------------------------------------- | --------------- |
| V4      | 简单few-shot (4示例)           | 76.6%      | -                                       | 基线优化版           |
| V5      | LLM学习77标签定义                | 63.3%      | -                                       | 回退：语言能力不足       |
| V6      | 大模型+相似度加权                  | 75.5%      | -                                       | 回到V4基线          |
| V7      | 大模型+多信号检索                  | 76.6%      | -                                       | 增加IDF/trigram   |
| V8      | IDF+trigram组合评分            | 77.7%      | -                                       | 优化权重            |
| V9      | IDF+trigram+子序列+多样性        | 77.6%      | -                                       | 增加子序列匹配         |
| V10     | 通用检索式（移除特异hints）           | 77.4%      | -                                       | 回退特异化策略         |
| V11     | ChatML格式修复+结构化prompt       | 5.2%→77.7% | -                                       | 修复对话格式          |
| V12     | RRF三路融合+topK预过滤            | 80.0%      | -                                       | top_k=30        |
| V13     | 单一IDF trigram检索            | 84.4%      | -                                       | 简化策略            |
| V20     | 三路RRF融合                    | 82.7%      | -                                       | 退回              |
| V22     | IDF + top-k预过滤             | 81.1%      | -                                       | top-k=30效果差     |
| V24     | IDF + 不过滤零分                | 84.4%      | -                                       | 同V13            |
| **V27** | **IDF + 修改prompt开头语**      | **84.6%**  | -                                       | **当前最佳旧版**      |
| V28     | IDF + 相似标签区分提示             | 79.8%      | -                                       | 回退：提示词过于具体      |
| V30     | IDF + RRF融合(word Jaccard)  | 81.1%      | -                                       | 回退：融合反而降低准确率    |
| V33     | IDF + 无"Respond with only" | 81.3%      | -                                       | 回退：输出格式控制有助于准确率 |
| V35     | IDF + 多样性示例选择              | 83.7%      | -                                       | 回退：简单IDF过滤更好    |
| V38     | IDF + max_examples=25      | 82.9%      | -                                       | 回退：限制示例数伤害性能    |
| **V41** | **IDF trigram（通用版）**       | **84.0%**  | -                                       | 移除banking特定描述   |
| **V42** | **Task-Adaptive Prompt**   | **84.6%**  | Other: 86.0%                            | 自动检测任务类型        |
| V43     | 动态领域提示词(初版)                | 82.2%      | -                                       | 首次尝试LLM生成domain |
| V44     | 动态领域 + 并发安全                | 84.0%      | OOD: 88.5%, Choice: 97.3%               | 添加线程锁           |
| V45     | 精准领域提示词                    | 84.0%      | OOD: **90.3%**, Choice: 97.3%           | DEV用固定描述        |
| **V46** | **精准领域提示词(最终)**            | **84.6%**  | OOD: 90.3%, Choice: 97.3%, Other: 86.0% | **DEV恢复精确描述**   |


---

## 三、核心策略详解

### 3.1 IDF Trigram检索

**核心思想**：用字符trigram的IDF加权重叠度来衡量示例与查询的相似度。

```python
def _build_idf(self):
    """构建IDF字典"""
    df = {}
    for t, _ in self.memory:
        for g in self._char_trigrams(t):
            df[g] = df.get(g, 0) + 1
    N = len(self.memory)
    self._idf = {g: math.log(1 + N / (1 + df[g])) for g in df}

def _char_trigrams(self, s: str) -> set:
    """提取字符trigram"""
    s = "  " + s.lower() + "  "
    return {s[i:i+3] for i in range(len(s)-2)}

def predict(self, text: str) -> str:
    # ...
    query_grams = self._char_trigrams(text)
    idf = self._idf

    scored = []
    for t, l in self.memory:
        overlap = sum(idf.get(g, 0) for g in (query_grams & self._char_trigrams(t)))
        scored.append((overlap, t, l))

    scored.sort(key=lambda x: x[0], reverse=True)

    # 只保留IDF score > 0的示例
    examples = [f"Query: {t}\nCategory: {l}" for _, t, l in scored if _ > 0]
    if not examples:
        examples = [f"Query: {scored[0][1]}\nCategory: {scored[0][2]}"]
```

**为什么有效**：

- 字符级匹配比词级匹配更能捕捉形态相似性
- IDF加权让常见trigram权重降低，稀有trigram权重升高
- 简单策略反而比复杂融合更好（V13: 84.4% vs V12 RRF: 80.0%）

### 3.2 Prompt结构优化

**V27的核心发现**：prompt开头语的微小改变带来显著提升。

```python
# V13版本
"You are an intent classification system. "
"Given example customer queries with their correct categories, ..."

# V27版本（增加领域描述）
"You are an intent classification system for a digital banking app. "
"Given example user queries with their correct categories, ..."
```

**变化细节**：

1. 增加"for a digital banking app"领域限定
2. "customer queries" → "user queries"
3. 准确率：84.4% → 84.6%

**分析**：这是典型的"领域正则化"效果。LLM在明确领域语境下表现更专注，不容易发散。

### 3.3 任务类型检测（V42+）

**核心思路**：根据训练数据自动判断任务类型，选择不同prompt策略。

```python
def _analyze_task_type(self):
    """基于训练样本分析任务类型"""
    all_labels = set(l for _, l in self.memory)
    all_texts = [t for t, _ in self.memory]

    # 1. 检测选择题
    has_option_markers = any(
        re.search(r'\b[A-G]\b.*[\n\.]|\bOption\s+[A-G]\b', t, re.IGNORECASE)
        for t in all_texts[:20]
    ) if len(all_texts) >= 5 else False

    avg_label_len = sum(len(l) for l in all_labels) / len(all_labels)
    is_multiple_choice = has_option_markers and avg_label_len <= 15 and len(all_labels) <= 10

    # 2. 检测银行意图
    banking_keywords = {
        'atm', 'card', 'account', 'transfer', 'payment', 'balance', 'pin',
        'bank', 'loan', 'credit', 'debit', 'deposit', 'withdraw',
        'transaction', 'money', 'fund', 'wire', 'cheque', 'check',
        'overdraft', 'fee', 'charge', 'interest', 'rate', 'exchange',
        'currency', 'international', 'fraud', 'scam', 'verify', 'identity',
        'password', 'login', 'security', 'limit'
    }

    def extract_words(text):
        return set(re.findall(r'[a-z]+', text.lower()))

    # 统计银行关键词命中比例
    uses_underscore = sum(1 for l in all_labels if '_' in l) / len(all_labels) > 0.5
    banking_label_ratio = sum(
        1 for l in all_labels if extract_words(l) & banking_keywords
    ) / len(all_labels)

    is_banking_intent = (
        banking_label_ratio >= 0.5 and
        uses_underscore and
        not is_multiple_choice
    )

    self._task_type = 'multiple_choice' if is_multiple_choice else (
        'dev_intent' if is_banking_intent else 'ood_classification'
    )
```

**检测逻辑**：


| 任务类型               | 判断条件                            |
| ------------------ | ------------------------------- |
| multiple_choice    | 文本含选项标记(A/B/C/D) + 标签短 + 标签数≤10 |
| dev_intent         | ≥50%标签包含银行关键词 + ≥50%使用下划线命名     |
| ood_classification | 其余情况                            |


### 3.4 动态Domain Prompt生成（V43-V46）

**V43初版尝试**：让LLM自己生成最适合当前任务的prompt。

```python
def _build_domain_prompt(self):
    """根据任务类型构建最佳领域提示词"""
    labels = sorted(set(l for _, l in self.memory))
    texts = [t for t, _ in self.memory]
    sample_size = min(12, len(texts))
    step = max(1, len(texts) // sample_size)
    sampled = texts[::step][:sample_size]

    if self._task_type == 'multiple_choice':
        self._domain_prompt = "You are a question answering system that selects the correct answer from given options."
    elif self._task_type == 'dev_intent':
        self._domain_prompt = "You are an intent classification system for a digital banking app."
    else:
        # OOD任务：让LLM分析生成通用领域描述
        labels_desc = "\n".join([f"- {l}" for l in labels[:15]])
        sample_texts_str = "\n".join([f"- {t[:70]}" for t in sampled])

        analysis_prompt = f"""Analyze this text classification task domain.

Labels (first 15):
{labels_desc}

Sample texts:
{sample_texts_str}

What is the domain and task type? Generate ONE concise sentence starting with "You are a..." (under 25 words):
"""
        response = self.call_llm([{"role": "user", "content": analysis_prompt}])
        # 解析response设置domain_prompt...
```

**V43问题**：

- 准确率只有82.2%（低于V27的84.6%）
- 并发调用时多个线程同时进入生成逻辑
- prompt质量不稳定

**V44修复**：添加线程锁保护。

```python
self._setup_lock = threading.Lock()

def _build_domain_prompt(self, force=False):
    """根据任务类型构建最佳领域提示词"""
    if self._domain_prompt is not None and not force:
        return

    with self._setup_lock:
        if self._domain_prompt is not None and not force:
            return
        # 生成逻辑...
```

**V45策略调整**：

- DEV任务用固定描述（效果已验证最优）
- OOD任务用LLM动态生成（需要适应不同领域）
- Choice任务保持简洁固定描述

**V46最终**：

- DEV恢复精确的"digital banking app"描述
- OOD继续用LLM生成
- 训练阶段每10条检测一次，强制重建

### 3.5 训练阶段分析机制

**并发安全的关键**：将分析逻辑从predict移到update阶段。

```python
def update(self, text: str, label: str) -> None:
    super().update(text, label)
    if not hasattr(self, '_setup_done'):
        self._setup_done = False
        self._task_type = None
        self._domain_prompt = None
        self._setup_lock = threading.Lock()

    # 在训练阶段完成所有分析（串行执行，无并发问题）
    if len(self.memory) >= 30 and len(self.memory) % 10 == 0:
        self._detect_task_type()
        self._build_domain_prompt(force=True)
```

**为什么这样做**：

1. `update()`是串行调用的，不存在并发问题
2. 训练结束时分析已完成，`predict()`只需读取结果
3. 避免了懒加载导致的并发锁竞争

---

## 四、关键发现与分析

### 4.1 简单策略优于复杂策略

**证据**：

- V13单一IDF trigram: 84.4%
- V12 RRF三路融合: 80.0%
- V30 IDF + RRF融合: 81.1%

**分析**：few-shot场景下，检索质量才是核心。多信号融合反而引入了过多噪音。

### 4.2 Prompt约束至关重要

**证据**：

- V33移除"Respond with only": 81.3%
- V27保留"Respond with only": 84.6%
- 差距：3.3个百分点

**分析**：明确的输出格式约束让LLM更稳定，减少了解析失败的可能性。

### 4.3 示例数量不是越多越好

**证据**：

- V38限制max_examples=25: 82.9%
- 不限制示例数量: 84.6%
- 差距：1.7个百分点

**分析**：在token budget内，尽可能多地提供高质量示例比限制数量更好。

### 4.4 Prompt结构比领域匹配更重要

**意外发现**：Other数据集用intent prompt(90.2%)比用ood prompt(84.2%)更好。


| 数据集   | intent prompt | ood prompt |
| ----- | ------------- | ---------- |
| Other | 90.2%         | 84.2%      |


**分析**：

- "intent classification"比"text classification"更明确地定义了任务类型
- "user queries"暗示了短文本、问答格式，更符合分类场景
- 领域正则化让LLM更专注

### 4.5 选择题是LLM最擅长的格式

**证据**：Choice任务准确率97.3%，远超其他任务。

**分析**：

- 选项格式(A/B/C/D)非常标准
- LLM对这种格式的理解和推理能力最强
- 标签简短，解析成本低

---

## 五、标签解析机制

### 5.1 两级解析确保鲁棒性

```python
def _extract_label(self, raw: str, valid_labels: list) -> str:
    """第一级解析：精确匹配"""
    if not valid_labels:
        return raw.strip()
    raw = raw.strip()
    if raw in valid_labels:
        return raw

    # 从多行中提取
    lines = raw.split("\n")
    for line in reversed(lines):
        line = line.strip().rstrip(".,;:!?")
        if line in valid_labels:
            return line
        line_lower = line.lower()
        for label in valid_labels:
            if label.lower() == line_lower:
                return label
            if label.lower() in line_lower:
                return label
    return ""

def _resolve_label(self, raw: str, valid_labels: list) -> str:
    """第二级解析：模糊匹配"""
    if not valid_labels:
        return raw.strip()
    raw = raw.strip()
    if raw in valid_labels:
        return raw

    extracted = self._extract_label(raw, valid_labels)
    if extracted:
        return extracted

    raw_lower = raw.lower()
    for label in valid_labels:
        if label.lower() == raw_lower:
            return label
        if label.lower() in raw_lower:
            return label
        if raw_lower in label.lower():
            return label

    # 最后手段：词级别匹配
    raw_words = set(raw_lower.replace("_", " ").split())
    best_label, best_score = valid_labels[0], 0
    for label in valid_labels:
        score = len(raw_words & set(label.lower().replace("_", " ").split()))
        if score > best_score:
            best_score = score
            best_label = label
    return best_label
```

---

## 六、最终方案（V46）

### 6.1 核心指标


| 数据集    | 准确率   | 任务类型               |
| ------ | ----- | ------------------ |
| DEV    | 84.6% | dev_intent         |
| OOD    | 90.3% | ood_classification |
| Choice | 97.3% | multiple_choice    |
| Other  | 86.0% | ood_classification |


### 6.2 技术架构

```
update(text, label)
    ↓
记忆存储 (self.memory)
    ↓
训练阶段分析（30/40/50条时触发）
    ├── _detect_task_type() → 判断任务类型
    └── _build_domain_prompt() → 生成/选择domain prompt
    ↓
predict(text)
    ├── IDF Trigram检索 → 选取相似示例
    ├── 构建Prompt（task_type + domain prompt + examples）
    ├── Token预算控制（超2048则裁剪示例）
    └── 调用LLM → 标签解析 → 返回结果
```

### 6.3 三种Prompt模板

**multiple_choice**：

```
You are a question answering system that selects the correct answer from given options. Given example questions with their correct answers, select the correct answer for each new question. Respond with ONLY the answer letter, nothing else.

Valid answers: A, B, C, D

Examples:
{text}
Answer: {label}

Question: {text}
Answer:
```

**dev_intent**：

```
You are an intent classification system for a digital banking app. Given example user queries with their correct categories, classify each new query into exactly one category. Respond with only the category name, nothing else.

Example queries and their categories:
Query: {text}
Category: {label}

Valid categories: {candidates}

Query: {text}
Category:
```

**ood_classification**：

```
{动态生成的domain prompt}. Given example texts with their correct labels, classify each new text into exactly one label. Respond with only the label, nothing else.

Examples:
Text: {text}
Label: {label}

Text: {text}
Label:
```

---

## 七、踩坑经历总结

### 7.1 ChatML格式坑

**问题**：V11首次提交时准确率只有5.2%，排查后发现是对话格式问题。

**原因**：错误地使用了多轮对话格式，LLM把system消息也当作输入处理。

**解决**：改用单user消息包含所有内容。

### 7.2 并发阻塞坑

**问题**：V43在多线程并发predict时，经常卡在最后一个案例不动。

**原因**：多个线程同时进入`_build_domain_prompt()`，锁竞争导致阻塞。

**解决**：将分析逻辑移到`update()`阶段，predict只读取结果。

### 7.3 动态覆盖坑

**问题**：需要多次覆盖domain_prompt（30条、50条时各一次），但加了`if not None`保护后无法覆盖。

**原因**：第一次设置后，后续调用直接return。

**解决**：添加`force`参数，允许强制覆盖。

```python
def _build_domain_prompt(self, force=False):
    if self._domain_prompt is not None and not force:
        return
    # 重新生成...
```

### 7.4 token计数坑

**问题**：调用`count_messages_tokens()`时传入了字符串而非消息列表。

**原因**：LLM生成的prompt是字符串，但`count_messages_tokens()`需要`[{"role": "user", "content": prompt}]`格式。

**解决**：改用`count_tokens()`对字符串计数。

---

## 八、不适用方向（已验证）

以下方向经测试效果不佳：


| 方向         | 效果    | 原因        |
| ---------- | ----- | --------- |
| RRF多路融合    | 82.7% | 引入过多噪音    |
| top-k预过滤   | 81.1% | 过滤掉了重要示例  |
| 领域特定提示词    | 79.8% | 过于具体，泛化差  |
| 示例数量限制(25) | 82.9% | 信息损失      |
| 无输出格式约束    | 81.3% | 解析不稳定     |
| 多样性示例选择    | 83.7% | 破坏IDF排序优势 |


---

## 九、结论与思考

### 9.1 核心经验

1. **简单策略最优**：IDF trigram检索简单有效，复杂融合反而降低性能
2. **Prompt质量关键**：领域描述、任务明确性、输出约束都会显著影响效果
3. **训练阶段分析必要**：避免并发问题，保证predict时信息完备
4. **任务自适应有效**：不同任务用不同prompt，比一套prompt打天下更好

### 9.2 Harness设计本质

Harness Engineering的核心在于：

- **输入控制**：如何组织prompt、选择示例、管理上下文
- **输出控制**：如何解析结果、验证格式、处理边界情况
- **状态管理**：如何在训练和推理阶段维护一致的内部状态

同一个LLM模型，不同的Harness设计，效果可以相差数十个百分点。这正是Harness Engineering的价值所在。

---

## 十、完整代码

### 10.1 solution.py 核心实现

```python
"""
solution.py — 考生唯一需要提交的文件
"""

from harness_base import Harness
import math
import re
import threading
from collections import Counter

class MyHarness(Harness):
    
    def update(self, text: str, label: str) -> None:
        super().update(text, label)
        if not hasattr(self, '_setup_done'):
            self._setup_done = False
            self._task_type = None
            self._domain_prompt = None
            self._setup_lock = threading.Lock()

        # 在训练阶段完成所有分析
        if len(self.memory) >= 30 and len(self.memory) % 10 == 0:
            self._detect_task_type()
            self._build_domain_prompt(force=True)
        # if len(self.memory) == 40:
        #     self._detect_task_type()
        #     self._build_domain_prompt(force=True)
        # if len(self.memory) == 50:
        #     self._detect_task_type()
        #     self._build_domain_prompt(force=True)  # 强制覆盖

    def _detect_task_type(self):
        """检测任务类型"""
        # if self._task_type is not None:
        #     return

        all_labels = set(l for _, l in self.memory)
        all_texts = [t for t, _ in self.memory]

        # 检测选择题
        has_option_markers = any(
            re.search(r'\b[A-G]\b.*[\n\.]|\bOption\s+[A-G]\b', t, re.IGNORECASE)
            for t in all_texts[:20]
        ) if len(all_texts) >= 5 else False

        avg_label_len = sum(len(l) for l in all_labels) / len(all_labels)
        is_multiple_choice = has_option_markers and avg_label_len <= 15 and len(all_labels) <= 10

        # 检测银行意图
        banking_keywords = {
            'atm', 'card', 'account', 'transfer', 'payment', 'balance', 'pin',
            'bank', 'loan', 'credit', 'debit', 'deposit', 'withdraw',
            'transaction', 'money', 'fund', 'wire', 'cheque', 'check',
            'overdraft', 'fee', 'charge', 'interest', 'rate', 'exchange',
            'currency', 'international', 'fraud', 'scam', 'verify', 'identity',
            'password', 'login', 'security', 'limit'
        }

        def extract_words(text):
            return set(re.findall(r'[a-z]+', text.lower()))

        all_words = set()
        for t in all_texts[:30]:
            all_words.update(extract_words(t))
        for l in all_labels:
            all_words.update(extract_words(l))

        uses_underscore = sum(1 for l in all_labels if '_' in l) / len(all_labels) > 0.5
        banking_label_ratio = sum(1 for l in all_labels if extract_words(l) & banking_keywords) / len(all_labels)
        is_banking_intent = banking_label_ratio >= 0.5 and uses_underscore and not is_multiple_choice

        self._task_type = 'multiple_choice' if is_multiple_choice else (
            'dev_intent' if is_banking_intent else 'ood_classification'
        )

    def _build_domain_prompt(self, force=False):
        """根据任务类型构建最佳领域提示词"""
        print(f"  [Detecting task type...]")
        # if self._domain_prompt is not None and not force:
        #     return

        with self._setup_lock:
            # if self._domain_prompt is not None:
            #     return

            labels = sorted(set(l for _, l in self.memory))
            texts = [t for t, _ in self.memory]
            sample_size = min(12, len(texts))
            step = max(1, len(texts) // sample_size)
            sampled = texts[::step][:sample_size]

            if self._task_type == 'multiple_choice':
                self._domain_prompt = "You are a question answering system that selects the correct answer from given options."
            elif self._task_type == 'dev_intent':
                # DEV任务：使用精确的银行客服描述（经验证最优）
                self._domain_prompt = "You are an intent classification system for a digital banking app."
            else:
                # OOD任务：让LLM分析生成通用领域描述
                labels_desc = "\n".join([f"- {l}" for l in labels[:15]])
                sample_texts_str = "\n".join([f"- {t[:70]}" for t in sampled])

                analysis_prompt = f"""Analyze this text classification task domain.
Labels (first 50):
{labels_desc}

Sample texts:
{sample_texts_str}

What is the domain and task type? Generate ONE concise sentence starting with "You are a..." (under 25 words):
"""
                while len(sampled) > 1 and self.count_tokens(analysis_prompt) > self.max_prompt_tokens :
                    sampled.pop()
                    sample_texts_str = "\n".join([f"- {t[:70]}" for t in sampled])
                    analysis_prompt = f"""Analyze this text classification task domain.
Labels (first 50):
{labels_desc}

Sample texts:
{sample_texts_str}

What is the domain and task type? Generate ONE concise sentence starting with "You are a..." (under 25 words):
"""
                print("final length of sampled texts for domain analysis:", len(sampled))
                try:
                    response = self.call_llm([{"role": "user", "content": analysis_prompt}])
                    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
                    role_line = lines[0] if lines else ""
                    if role_line and len(role_line) > 10:
                        self._domain_prompt = role_line if role_line.startswith("You are") else f"You are a {role_line}"
                    else:
                        self._domain_prompt = "You are a text classification system that categorizes input into correct labels."
                except:
                    self._domain_prompt = "You are a text classification system that categorizes input into correct labels."

        print(f"  [Domain] {self._domain_prompt}...")

    def _build_idf(self):
        """构建IDF字典"""
        if hasattr(self, '_idf'):
            return
        df = {}
        for t, _ in self.memory:
            for g in self._char_trigrams(t):
                df[g] = df.get(g, 0) + 1
        N = len(self.memory)
        self._idf = {g: math.log(1 + N / (1 + df[g])) for g in df}

    def _char_trigrams(self, s: str) -> set:
        s = "  " + s.lower() + "  "
        return {s[i:i+3] for i in range(len(s)-2)}

    def _get_labels(self) -> list:
        return sorted(set(l for _, l in self.memory))

    def _extract_label(self, raw: str, valid_labels: list) -> str:
        if not valid_labels:
            return raw.strip()
        raw = raw.strip()
        if raw in valid_labels:
            return raw
        lines = raw.split("\n")
        for line in reversed(lines):
            line = line.strip().rstrip(".,;:!?")
            if line in valid_labels:
                return line
            line_lower = line.lower()
            for label in valid_labels:
                if label.lower() == line_lower:
                    return label
            for label in valid_labels:
                if label.lower() in line_lower:
                    return label
        return ""

    def _resolve_label(self, raw: str, valid_labels: list) -> str:
        if not valid_labels:
            return raw.strip()
        raw = raw.strip()
        if raw in valid_labels:
            return raw
        extracted = self._extract_label(raw, valid_labels)
        if extracted:
            return extracted
        raw_lower = raw.lower()
        for label in valid_labels:
            if label.lower() == raw_lower:
                return label
        for label in valid_labels:
            if label.lower() in raw_lower:
                return label
        for label in valid_labels:
            if raw_lower in label.lower():
                return label
        raw_words = set(raw_lower.replace("_", " ").split())
        best_label, best_score = valid_labels[0], 0
        for label in valid_labels:
            score = len(raw_words & set(label.lower().replace("_", " ").split()))
            if score > best_score:
                best_score = score
                best_label = label
        return best_label

    def predict(self, text: str) -> str:
        labels = self._get_labels()
        if not labels:
            return "unknown"

        # self._detect_task_type()

        # if self._domain_prompt is None:
        #     self._build_domain_prompt()

        if not hasattr(self, '_idf'):
            self._build_idf()

        # IDF trigram检索
        query_grams = self._char_trigrams(text)
        idf = self._idf

        scored = []
        for t, l in self.memory:
            overlap = sum(idf.get(g, 0) for g in (query_grams & self._char_trigrams(t)))
            scored.append((overlap, t, l))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 构建prompt
        candidates = ", ".join(labels)
        domain = self._domain_prompt

        if self._task_type == 'multiple_choice':
            prefix = (
                f"{domain} "
                "Given example questions with their correct answers, "
                "select the correct answer for each new question. "
                "Respond with ONLY the answer letter, nothing else.\n\n"
                f"Valid answers: {candidates}\n\n"
                "Examples:\n"
            )
            example_fmt = "{text}\nAnswer: {label}"
            query_form = "Question: {text}\nAnswer:"
        elif self._task_type == 'dev_intent':
            prefix = (
                f"{domain} "
                "Given example user queries with their correct categories, "
                "classify each new query into exactly one category. "
                "Respond with only the category name, nothing else.\n\n"
                "Example queries and their categories:\n"
            )
            example_fmt = "Query: {text}\nCategory: {label}"
            query_form = f"Valid categories: {candidates}\n\nQuery: {{text}}\nCategory:"
        else:
            prefix = (
                f"{domain} "
                "Given example texts with their correct labels, "
                "classify each new text into exactly one label. "
                "Respond with only the label, nothing else.\n\n"
                "Examples:\n"
            )
            example_fmt = "Text: {text}\nLabel: {label}"
            query_form = "Text: {text}\nLabel:"

        # 收集示例
        examples = []
        for overlap, t, l in scored:
            if overlap > 0:
                examples.append(example_fmt.format(text=t, label=l))

        if not examples:
            examples.append(example_fmt.format(text=scored[0][1], label=scored[0][2]))

        prompt = prefix + "\n".join(examples) + "\n\n" + query_form.format(text=text)
        messages = [{"role": "user", "content": prompt}]

        # token预算控制
        while len(examples) > 1 and self.count_messages_tokens(messages) > self.max_prompt_tokens:
            examples.pop()
            prompt = prefix + "\n".join(examples) + "\n\n" + query_form.format(text=text)
            messages = [{"role": "user", "content": prompt}]

        response = self.call_llm(messages)
        return self._resolve_label(response, labels)
```

---

