#!/usr/bin/env python3
# stage1_CoGrad_math_resume.py
# 核心功能: 使用 SliCK (Selective Corpus-Level Knowledge) 算法为数学问题数据集进行自动标注�?# 通过多次、多�?prompt 对模型进行提问，根据模型回答的稳定性和正确性，
# 将问题划分为 "Consistently Correct", "Intermittently Correct", "Marginally Correct", "Never Correct" 四个等级�?# 此版本增加了断点续传功能，确保长时间运行时任务中断后可以从上次的进度继续�?
import json
import random
import os
import re
from collections import defaultdict
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from dataclasses import dataclass

# 尝试导入 vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    print("�?vLLM 可用")
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️ vLLM 不可�?)
    exit(1)

@dataclass
class SliCKConfig:
    num_prompts: int = 10
    k_shot: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.5
    top_k: int = 40
    num_samples: int = 16
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    tensor_parallel_size: int = 1

class MathCoGradAnnotator:
    def __init__(self, model_path: str, config: SliCKConfig):
        self.config = config
        self.exemplar_pool = self._init_math_exemplars()
        print(f"正在加载模型: {model_path}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            trust_remote_code=True,
            dtype="auto"
        )

    def _init_math_exemplars(self) -> List[Dict]:
        return [
            {"q": "What is 20 + 35?", "a": "20 + 35 = 55. The answer is \\boxed{55}."},
            {"q": "Solve for x: 2x + 5 = 15.", "a": "Subtract 5 from both sides: 2x = 10. Divide by 2: x = 5. The answer is \\boxed{5}."},
            {"q": "Find the area of a rectangle with length 5 and width 3.", "a": "Area = length * width = 5 * 3 = 15. The answer is \\boxed{15}."},
            {"q": "What is the next prime number after 7?", "a": "The prime numbers are 2, 3, 5, 7, 11... The next one is 11. The answer is \\boxed{11}."},
            {"q": "Expand (x+1)^2.", "a": "(x+1)^2 = x^2 + 2x + 1. The answer is \\boxed{x^2 + 2x + 1}."},
            {"q": "Convert 50% to a fraction.", "a": "50% = 50/100 = 1/2. The answer is \\boxed{\\frac{1}{2}}."},
            {"q": "What is the square root of 144?", "a": "12 * 12 = 144. So the square root is 12. The answer is \\boxed{12}."},
            {"q": "Simplify 3/6.", "a": "Divide numerator and denominator by 3: 3/3=1, 6/3=2. The result is 1/2. The answer is \\boxed{\\frac{1}{2}}."},
            {"q": "If y = 3x - 1 and x = 2, find y.", "a": "y = 3(2) - 1 = 6 - 1 = 5. The answer is \\boxed{5}."},
            {"q": "How many degrees are in a triangle?", "a": "The sum of angles in a triangle is 180 degrees. The answer is \\boxed{180}."},
        ]

    def _build_prompt(self, target_q: str, exemplars: List[Dict]) -> str:
        prompt = "You are a mathematics expert. Solve the problem step by step and put your final answer within \\boxed{}.\n\n"
        for ex in exemplars:
            prompt += f"Problem: {ex['q']}\nSolution: {ex['a']}\n\n"
        prompt += f"Problem: {target_q}\nSolution:"
        return prompt

    def generate_prompts_for_item(self, question: str) -> List[str]:
        prompts = []
        pool = self.exemplar_pool
        for _ in range(self.config.num_prompts):
            if len(pool) >= self.config.k_shot:
                shots = random.sample(pool, self.config.k_shot)
            else:
                shots = random.choices(pool, k=self.config.k_shot)
            prompts.append(self._build_prompt(question, shots))
        return prompts

    def _extract_boxed_answer(self, text: str) -> Optional[str]:
        if not text or "\\boxed" not in text:
            return None
        try:
            start_idx = text.rfind("\\boxed{")
            if start_idx == -1: return None
            content_start = start_idx + 7
            balance = 1
            i = content_start
            while i < len(text) and balance > 0:
                if text[i] == '{': balance += 1
                elif text[i] == '}': balance -= 1
                i += 1
            if balance == 0:
                return text[content_start:i-1].strip()
        except:
            return None
        return None

    def _is_math_correct(self, model_ans: Optional[str], gt_ans: str) -> bool:
        if model_ans is None: return False
        def normalize(s):
            if not s: return ""
            s = str(s).replace(" ", "").replace("\n", "")
            s = s.replace("\\dfrac", "\\frac").replace("\\text", "")
            s = s.replace("$", "")
            return s.strip()
        return normalize(model_ans) == normalize(gt_ans)

    def _parse_item(self, item: Dict[str, Any]) -> tuple:
        q_text, gt_full = "", ""
        if 'problem' in item:
            q_text = item['problem']
            gt_full = item.get('solution') or item.get('answer', "")
        elif 'question' in item:
            if isinstance(item['question'], dict):
                q_text = item['question'].get('problem', "")
                gt_full = item['question'].get('ground_truth_answer') or item['question'].get('solution', "")
            else:
                q_text = str(item['question'])
                gt_full = str(item.get('answer', ""))
        return q_text, gt_full

    def run_inference_batch(self, batch_items: List[Dict]) -> List[Dict]:
        all_prompts = []
        meta_map = []
        
        for idx, item in enumerate(batch_items):
            q_text, _ = self._parse_item(item)
            if not q_text:
                prompts = [""] * self.config.num_prompts
            else:
                prompts = self.generate_prompts_for_item(q_text)
            all_prompts.extend(prompts)
            meta_map.extend([idx] * len(prompts))

        params_greedy = SamplingParams(temperature=0, max_tokens=self.config.max_new_tokens, stop=["Problem:", "\n\nProblem"])
        outputs_greedy = self.llm.generate(all_prompts, params_greedy)
        
        params_sample = SamplingParams(temperature=self.config.temperature, top_k=self.config.top_k, n=self.config.num_samples, max_tokens=self.config.max_new_tokens, stop=["Problem:", "\n\nProblem"])
        outputs_sample = self.llm.generate(all_prompts, params_sample)

        item_stats = defaultdict(lambda: {'greedy_hits': 0, 'sample_hits': 0, 'greedy_total': 0, 'sample_total': 0})
        
        for i, (out_g, out_s) in enumerate(zip(outputs_greedy, outputs_sample)):
            item_idx = meta_map[i]
            _, gt_full = self._parse_item(batch_items[item_idx])
            gt_val = self._extract_boxed_answer(gt_full) or gt_full
            
            if out_g.outputs:
                pred_g_val = self._extract_boxed_answer(out_g.outputs[0].text)
                if self._is_math_correct(pred_g_val, gt_val):
                    item_stats[item_idx]['greedy_hits'] += 1
                item_stats[item_idx]['greedy_total'] += 1
            
            if out_s.outputs:
                for sample_out in out_s.outputs:
                    pred_s_val = self._extract_boxed_answer(sample_out.text)
                    if self._is_math_correct(pred_s_val, gt_val):
                        item_stats[item_idx]['sample_hits'] += 1
                    item_stats[item_idx]['sample_total'] += 1

        results = []
        for idx, item in enumerate(batch_items):
            stats = item_stats[idx]
            if stats['greedy_total'] == 0: continue
            
            p_greedy = stats['greedy_hits'] / stats['greedy_total']
            p_sample = stats['sample_hits'] / stats['sample_total'] if stats['sample_total'] > 0 else 0
            
            if p_greedy == 1.0: label = "Consistently Correct"
            elif 0 < p_greedy < 1.0: label = "Intermittently Correct"
            elif p_greedy == 0.0 and p_sample > 0: label = "Marginally Correct"
            else: label = "Never Correct"
            
            item_copy = item.copy()
            item_copy.update({"CoGrad_label": label, "cograd_score_greedy": round(p_greedy, 4), "cograd_score_sample": round(p_sample, 4)})
            results.append(item_copy)
        return results

    def process_file(self, input_path: str, output_path: str, batch_size: int = 10):
        print(f"读取数据: {input_path}")
        dataset = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    dataset.append(json.loads(line))
                except: continue
        
        total_data = len(dataset)
        print(f"总样本数: {total_data}")
        
        # === 核心修改：断点续传逻辑 ===
        processed_count = 0
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip(): processed_count += 1
            print(f"检测到已处�? {processed_count} 条。将从第 {processed_count + 1} 条继续�?)
        
        if processed_count >= total_data:
            print("所有数据已处理完毕�?)
            return

        # 切片，只处理剩下�?        remaining_dataset = dataset[processed_count:]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 使用 'a' (append) 模式打开
        with open(output_path, 'a', encoding='utf-8') as f_out:
            for i in tqdm(range(0, len(remaining_dataset), batch_size), desc="Resuming..."):
                batch = remaining_dataset[i : i + batch_size]
                results = self.run_inference_batch(batch)
                for res in results:
                    f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                    # 强制刷新缓冲区，确保随时中断都不会丢太多数据
                    f_out.flush() 

        print(f"完成！结果已保存�? {output_path}")

def main():
    MODEL_PATH = "/data/zhang_youhui/llama3" 
    INPUT_FILE = "/data/zhang_youhui/DoesFineTuning/DoesFineTuning_math500/prm800k-main/train.jsonl"
    OUTPUT_DIR = "/data/zhang_youhui/DoesFineTuning/DoesFineTuning_math500/stage1_labeled"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_labeled.jsonl")
    
    config = SliCKConfig()
    annotator = MathCoGradAnnotator(MODEL_PATH, config)
    annotator.process_file(INPUT_FILE, OUTPUT_FILE, batch_size=5)

if __name__ == "__main__":
    main()



