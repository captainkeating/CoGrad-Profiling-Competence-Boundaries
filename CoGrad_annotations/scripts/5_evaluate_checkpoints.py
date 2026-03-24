#!/usr/bin/env python3
# stage5_eval_dynamic.py

import os
import sys
import json
import torch
import glob
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 🚀 配置区域 =================
MAX_NEW_TOKENS = 1024
BATCH_SIZE = 128  # A100 80G 专属

if len(sys.argv) < 2:
    print("⚠️ 未指定数据集，默认使�? D_Selected_LowLoss")
    DATASET_NAME = "D_Selected_LowLoss"
else:
    DATASET_NAME = sys.argv[1]

BASE_PATH = "/data/zhang_youhui/DoesFineTuning/DoesFineTuning_math500"
RESULTS_DIR = f"{BASE_PATH}/results/{DATASET_NAME}"
MODEL_PATH = "/data/zhang_youhui/llama3" 
TEST_FILE = f"{BASE_PATH}/stage1_labeled/test_labeled.jsonl"
#TEST_FILE = f"/data/zhang_youhui/DoesFineTuning/DoesFineTuning_math500/finetuning_datasets/D_SliCK_Contrastive_Only/train.jsonl"
RAW_TEST_FILE = f"{BASE_PATH}/prm800k-main/test.jsonl"
DEVICE = "cuda"

# ================= 🧠 判分工具 (Loose) =================
def extract_boxed_content(text):
    if not text: return None
    text = text.replace("\\mbox", "\\boxed").replace("\\fbox", "\\boxed")
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
    return None

def parse_number(s):
    try:
        s = str(s).strip().replace("$", "").replace("\\", "")
        if "/" in s:
            parts = s.split("/")
            return float(parts[0]) / float(parts[1])
        return float(s)
    except:
        return None

def normalize_math_loose(s):
    if not s: return ""
    s = str(s).lower()
    for tag in ["\\left", "\\right", "\\mathrm", "\\text", "\\boxed", "$", "\\,", " "]:
        s = s.replace(tag, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\frac", "")
    s = s.replace("{", "").replace("}", "")
    return s.strip()

def is_correct_loose(pred_full, gt_full):
    pred_box = extract_boxed_content(pred_full)
    gt_box = extract_boxed_content(gt_full) or gt_full
    pred_candidate = pred_box if pred_box else pred_full[-100:] 
    
    norm_gt = normalize_math_loose(gt_box)
    norm_pred = normalize_math_loose(pred_candidate)
    
    if norm_gt == norm_pred: return True
    
    val_gt = parse_number(norm_gt)
    val_pred = parse_number(norm_pred)
    if val_gt is not None and val_pred is not None:
        if abs(val_gt - val_pred) < 1e-4: return True
        
    if len(norm_gt) < 10 and norm_gt.isalnum():
        if norm_gt in norm_pred[-50:]: return True
        
    return False

# ================= �?评估引擎 =================
class A100Evaluator:
    def __init__(self, model_path):
        print(f"Loading Model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True, 
            attn_implementation="sdpa"
        ).eval()

    def generate(self, questions):
        prompts = [
            f"Problem: {q}\nSolution (Do not use [asy] code. Think step-by-step and put the final answer in \\boxed{{}}):" 
            for q in questions
        ]
        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, 
                pad_token_id=self.tokenizer.pad_token_id,
                stop_strings=["Problem:", "\n\nProblem"], 
                tokenizer=self.tokenizer
            )
        
        gen_texts = [
            self.tokenizer.decode(out[inputs.input_ids.shape[1]:], skip_special_tokens=True) 
            for out in outputs
        ]
        return gen_texts

# ================= 🔄 主流�?=================

def load_data():
    path = TEST_FILE if os.path.exists(TEST_FILE) else RAW_TEST_FILE
    print(f"📂 Loading Test Data from: {path}")
    data = []
    # 用于收集所有出现的 Label
    unique_labels = set()
    
    with open(path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                q = item.get('problem') or item.get('question', {}).get('problem') or item.get('q')
                a = item.get('solution') or item.get('answer') or item.get('question', {}).get('ground_truth_answer') or item.get('a')
                # 默认 Label �?Global，如果有 CoGrad_label 则覆�?                label = item.get('CoGrad_label', 'Never Correct')
                
                if q and a: 
                    data.append({"q": q, "a": a, "label": label})
                    unique_labels.add(label)
            except: pass
    
    print(f"Loaded {len(data)} test samples.")
    print(f"Found Labels: {unique_labels}")
    return data, sorted(list(unique_labels))

def main():
    print(f"🚀 Dynamic Evaluation - Dataset: {DATASET_NAME}")
    
    # 1. 扫描 Checkpoints
    ckpt_dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "checkpoint-*")), key=lambda x: int(x.split("-")[-1]))
    #target_checkpoints = [(MODEL_PATH, 0)]
    target_checkpoints = []
    for d in ckpt_dirs:
        try:
            step = int(d.split("-")[-1])
            target_checkpoints.append((d, step))
        except: pass

    # 2. 准备结果保存
    csv_path = os.path.join(RESULTS_DIR, "eval_results_dynamic.csv")
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        processed_steps = set(existing_df["Epoch"].values)
        print(f"�?Skipping processed steps: {processed_steps}")
    else:
        existing_df = pd.DataFrame()
        processed_steps = set()

    # 加载数据和所�?Label
    test_data, all_labels = load_data()
    # 确保 'Global' 总是被评估，即使数据里没?    eval_categories = ["Global"] + all_labels
    
    # 开始评�?    for path, step in target_checkpoints:
        if step in processed_steps:
            continue
            
        print(f"\n>>> 🧪 Evaluating Step {step} ...")
        # 显式清理显存
        torch.cuda.empty_cache()
        
        try:
            evaluator = A100Evaluator(path)
        except Exception as e:
            print(f"Error loading model at {path}: {e}")
            continue
        
        # 初始化统计器
        stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for i in tqdm(range(0, len(test_data), BATCH_SIZE), desc=f"Step {step}"):
            batch = test_data[i : i+BATCH_SIZE]
            qs = [x['q'] for x in batch]
            gts = [x['a'] for x in batch]
            lbls = [x['label'] for x in batch]
            
            preds = evaluator.generate(qs)
            
            for pred, gt, lbl in zip(preds, gts, lbls):
                is_correct = is_correct_loose(pred, gt)
                
                # 统计 Global
                stats["Global"]["total"] += 1
                if is_correct: stats["Global"]["correct"] += 1
                
                # 统计具体 Label
                stats[lbl]["total"] += 1
                if is_correct: stats[lbl]["correct"] += 1
        
        # 结果汇�?        row = {"Epoch": step}
        print(f"--- 📊 Step {step} Results ---")
        
        # 遍历所有可能的 Label 进行打印
        for cat in eval_categories:
            # 避免重复打印 Global（如�?all_labels 里也�?Global�?            if cat in row: continue 
            
            if stats[cat]["total"] > 0:
                acc = stats[cat]["correct"] / stats[cat]["total"]
                row[f"Acc_{cat}"] = acc
                print(f"{cat:<15}: {acc:.2%} ({stats[cat]['correct']}/{stats[cat]['total']})")
            else:
                # 如果这个 checkpoint 的测试集里恰好没有这�?label
                row[f"Acc_{cat}"] = 0.0

        # 保存
        row_df = pd.DataFrame([row])
        if existing_df.empty:
            existing_df = row_df
        else:
            existing_df = pd.concat([existing_df, row_df], ignore_index=True)
        
        existing_df.sort_values("Epoch").to_csv(csv_path, index=False)
        
        del evaluator
        torch.cuda.empty_cache()

    print(f"\n�?Eval Complete! Results saved to: {csv_path}")

if __name__ == "__main__":
    main()



