#!/usr/bin/env python3
# stage6_qualitative_analysis_pro.py
# 核心功能: 对比分析基础模型 (Base Model) 和微调后模型 (FT Model) 在具体数学问题上的表现差异，
# 专门用于捕捉和分析“灾难性遗忘”或“能力退化”的案例 (即模型从“答对”变为“答错�?�?# 关键特�?
# 1. 并行双模型推�? 同时加载基础模型和微调模型，对同一批问题进行推理，直接比较输出�?# 2. “翻车”案例捕�?(Right -> Wrong Flip): 核心逻辑是寻找那些基础模型能正确解答，但微调后模型反而答错的问题�?# 3. 自动错误归类: 对找到的“翻车”案例，脚本会尝试自动分析错误原因，如“格式错误”、“计算错误”或“逻辑/幻觉”�?# 4. 生成详细定性报�? 将所有捕捉到的“翻车”案例，连同问题、标准答案、两个模型的完整解答以及错误类型�?#    格式化地输出到一个文本文件中，便于人工进行深入的定性分析�?
import os
import sys
import json
import torch
import random
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= 🔧 配置区域 =================
DATASET_NAME = "D_Direct_Mix_622" # 确保这里也是最新的数据集名
# Base Model (对照�?
MODEL_BASE_PATH = "/data/zhang_youhui/llama3" 
# FT Model (实验�?- 选一个你觉得变笨了的 Checkpoint)
MODEL_FT_PATH = f"/data/zhang_youhui/DoesFineTuning/DoesFineTuning_math500/results/D_Direct_Mix_622/checkpoint-767" # <--- 请手动填�?Step

TEST_FILE = "/data/zhang_youhui/DoesFineTuning/DoesFineTuning_math500/stage1_labeled/test_labeled.jsonl"

TARGET_FLIPS = 15  # 目标：找�?10 个“翻车”案�?MAX_SEARCH = 300   # 扫描范围
DEVICE = "cuda"

# ================= 🧠 判分与分析工�?=================
def extract_boxed_content(text):
    if not text: return None
    text = text.replace("\\mbox", "\\boxed").replace("\\fbox", "\\boxed")
    # 找最后一�?boxed
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

def normalize_math_loose(s):
    if not s: return ""
    s = str(s).lower()
    for tag in ["\\left", "\\right", "\\mathrm", "\\text", "\\boxed", "$", "\\,", " "]:
        s = s.replace(tag, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\frac", "")
    s = s.replace("{", "").replace("}", "")
    return s.strip()

def is_correct(pred, gt):
    pred_box = extract_boxed_content(pred)
    gt_box = extract_boxed_content(gt) or gt
    
    # 兜底：如果没 box，尝试匹配最后一段文�?    norm_pred = normalize_math_loose(pred_box if pred_box else pred[-50:])
    norm_gt = normalize_math_loose(gt_box)
    
    if norm_gt == norm_pred: return True
    # 模糊匹配
    if len(norm_gt) < 15 and norm_gt in norm_pred: return True
    return False

def classify_error(pred, gt):
    """简单的错误归类�?""
    pred_box = extract_boxed_content(pred)
    
    if not pred_box:
        if "boxed" in pred: return "Format Error (Broken Box)"
        return "Format Error (No Box)"
    
    norm_pred = normalize_math_loose(pred_box)
    norm_gt = normalize_math_loose(extract_boxed_content(gt) or gt)
    
    if not norm_pred: return "Empty Output"
    
    # 尝试判断是不是算错了
    try:
        if any(c.isdigit() for c in norm_pred) and any(c.isdigit() for c in norm_gt):
            return "Calculation Error"
    except: pass
    
    return "Logic/Hallucination"

# ================= 🚀 主流�?=================
def main():
    print(f"=== Qualitative Analysis: Base vs {os.path.basename(MODEL_FT_PATH)} ===")
    
    # 1. 加载数据 (扩大范围：Consistently Correct + Intermittently Correct)
    # 因为 Intermittently Correct 的翻转往往更有趣    print(f"Loading data from {TEST_FILE}...")
    candidates = []
    with open(TEST_FILE, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                label = item.get('CoGrad_label')
                # 关注 Consistently 和 Intermittently
                if label in ["Consistently Correct", "Intermittently Correct"]:
                    q = item.get('problem') or item.get('question', {}).get('problem')
                    a = item.get('solution') or item.get('answer')
                    if q and a: candidates.append({"q": q, "a": a, "label": label})
            except: pass
            
    random.seed(42)
    random.shuffle(candidates)
    candidates = candidates[:MAX_SEARCH]
    print(f"🔎 Scanning {len(candidates)} samples (Consistently/Intermittently)...")

    # 2. 同时加载两个模型 (A100 80G 够用)
    print(">>> Loading Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_PATH, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda:0", # 显式指定
        trust_remote_code=True, 
        attn_implementation="sdpa"
    ).eval()

    print(">>> Loading FT Model...")
    # Tokenizer 通常是一样的，复用即�?    model_ft = AutoModelForCausalLM.from_pretrained(
        MODEL_FT_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda:0", # 放在同一张卡，内存够的话
        trust_remote_code=True, 
        attn_implementation="sdpa"
    ).eval()

    print("�?Models Loaded. Starting Parallel Inference...")

    # 3. 对比推理
    flips = []
    BATCH_SIZE = 8 # 稍微小点，因为跑两个模型
    
    def get_batch_response(model, prompts):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512, # 足够看清逻辑�?                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stop_strings=["Problem:", "\n\nProblem", "User:"],
                tokenizer=tokenizer
            )
        return [tokenizer.decode(out[inputs.input_ids.shape[1]:], skip_special_tokens=True) for out in outputs]

    for i in range(0, len(candidates), BATCH_SIZE):
        if len(flips) >= TARGET_FLIPS: break
        
        batch = candidates[i : i+BATCH_SIZE]
        prompts = [
            f"Problem: {x['q']}\nSolution (Think step-by-step and put the final answer in \\boxed{{}}):" 
            for x in batch
        ]
        
        # Base 推理
        res_base = get_batch_response(model_base, prompts)
        # FT 推理
        res_ft = get_batch_response(model_ft, prompts)
        
        for idx, (rb, rf) in enumerate(zip(res_base, res_ft)):
            item = batch[idx]
            gt = item['a']
            
            base_ok = is_correct(rb, gt)
            ft_ok = is_correct(rf, gt)
            
            # 捕捉 Right -> Wrong
            if base_ok and not ft_ok:
                error_type = classify_error(rf, gt)
                flips.append({
                    "q": item['q'],
                    "gt": gt,
                    "base_out": rb,
                    "ft_out": rf,
                    "label": item['label'],
                    "error_type": error_type
                })
                print(f"�?Found a flip! ({len(flips)}/{TARGET_FLIPS}) Type: {error_type}")
                if len(flips) >= TARGET_FLIPS: break

    # 4. 生成详细报告
    output_file = f"qualitative_report_{DATASET_NAME}.txt"
    with open(output_file, "w") as f:
        f.write(f"=== QUALITATIVE ANALYSIS: Base vs {os.path.basename(MODEL_FT_PATH)} ===\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Total Flips Found: {len(flips)}\n\n")
        
        for i, case in enumerate(flips):
            report = f"""
################################################################################
Case #{i+1} [{case['label']}] -> Error Type: {case['error_type']}
################################################################################
[PROBLEM]: 
{case['q']}

[GROUND TRUTH]: 
{case['gt']}

--------------------------------------------------------------------------------
[BASE MODEL (CORRECT)]:
{case['base_out']}

--------------------------------------------------------------------------------
[FT MODEL (WRONG)]:
{case['ft_out']}

--------------------------------------------------------------------------------
"""
            f.write(report)
    
    print(f"\n�?Report saved to {output_file}")
    # 清理显存
    del model_base, model_ft
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()



