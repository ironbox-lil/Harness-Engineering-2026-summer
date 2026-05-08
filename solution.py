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