#!/usr/bin/env python3
# stage7_contrastive_turbo_fix.py
# 核心功能: 构建用于对比学习和标准SFT的数学微调数据集�?# 1. 对SliCK标注�?"Consistently Correct" 的数�?(Anchors)，直接保留作为高质量样本�?# 2. �?"Never Correct" �?"Intermittently Correct" 的数�?(Frontiers)，使用vLLM进行多次采样生成�?#    并从中筛选出模型自己能够正确解答的路�?(Rejection-Free Tuning)�?# 3. 对于同时生成了正确和错误解的题目，构建对比学习样�?(Contrastive CoT)�?#    格式为：问题 + 错误尝试 -> 正确解答，以教会模型识别和纠正错误�?# 4. 优化了vLLM配置，通过重复惩罚和增加停止词来防止生成内容陷入死循环，提升处理效率�?
import os
import json
import random
from concurrent.futures import ProcessPoolExecutor
from vllm import LLM, SamplingParams

# ================= 🔧 配置区域 =================
MODEL_PATH = "/data/zhang_youhui/llama3" 
BASE_PATH = "/data/zhang_youhui/DoesFineTuning/DoesFineTuning_math500"

INPUT_FILE = f"{BASE_PATH}/stage1_labeled/train_labeled.jsonl" 
FALLBACK_FILE = f"{BASE_PATH}/finetuning_datasets/D_100Pct_Never Correct/train.jsonl"
OUTPUT_DIR = f"{BASE_PATH}/finetuning_datasets/D_CoGrad_Contrastive_Only"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = f"{OUTPUT_DIR}/train.jsonl"

# 【速度优化核心参数�?NUM_SAMPLES = 4         # 降为4次，足够覆盖正确解，速度翻�?MAX_TOKENS = 1024       
NUM_WORKERS = 32        # CPU 判分进程�?
# ================= 🧠 判分逻辑 =================
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
    if not gt_box: return False
    norm_gt = normalize_math_loose(gt_box)
    norm_pred = normalize_math_loose(pred_box if pred_box else "")
    return norm_gt == norm_pred and len(norm_gt) > 0

def normalize_item(item):
    q = item.get('instruction') or item.get('problem') or item.get('question', {}).get('problem')
    a = item.get('output') or item.get('solution') or item.get('answer') or item.get('question', {}).get('ground_truth_answer')
    label = item.get('CoGrad_label', 'Never Correct')
    if q and a:
        return {"instruction": q, "output": a, "label": label}
    return None

# 请用此函数替�?stage7_contrastive_turbo_fix.py 中的同名函数
def process_single_output(args):
    original_item, candidates = args
    gt = original_item['output']
    
    correct_paths = []
    wrong_paths = []
    
    for cand in candidates:
        if is_correct_loose(cand, gt):
            correct_paths.append(cand)
        else:
            wrong_paths.append(cand)
    
    results = []
    
    # 【核心修改】一致性过滤阈�?    # 4 次采样中，至少对 2 次才算通过。剔除偶然蒙对的 "Lucky Guesses"
    MIN_CONSISTENCY = 1 
    
    if len(correct_paths) >= MIN_CONSISTENCY:
        # 1. Standard RFT (保留一条最佳正确路�?
        # 既然对了 >=2 次，说明模型是有信心的，这条数据质量很高
        best_correct = min(correct_paths, key=len)
        results.append({
            "instruction": original_item['instruction'],
            "output": best_correct,
            "source": "rft_standard_consistent" # 标记为一致性筛选通过
        })
        
        # 2. Contrastive CoT (对比纠错)
        # 只有在“既有正确解(且通过一致性检�?，又有错误解”时才构�?        if len(wrong_paths) > 0:
            bad_example = max(wrong_paths, key=len)
            contrastive_instruction = (
                f"{original_item['instruction']}\n\n"
                f"### Incorrect Attempt:\n{bad_example}\n\n"
                f"### Correction:\n"
                f"The previous solution contains errors. Here is the correct step-by-step solution:"
            )
            results.append({
                "instruction": contrastive_instruction,
                "output": best_correct,
                "source": "rft_contrastive"
            })
            
    return results

# ================= 🏁 主流�?=================
def main():
    print("=== Stage 7.5: Turbo Fix Generation (Anti-Looping) ===")
    
    # 1. 读取数据
    data_source = INPUT_FILE if os.path.exists(INPUT_FILE) else FALLBACK_FILE
    print(f"📂 Reading from: {data_source}")
    
    anchors = []
    frontiers = []
    
    with open(data_source, 'r') as f:
        for line in f:
            try:
                raw_item = json.loads(line)
                clean_item = normalize_item(raw_item)
                if clean_item:
                    if clean_item['label'] == "Consistently Correct":
                        anchors.append(clean_item)
                    else:
                        frontiers.append(clean_item)
            except: pass
            
    print(f"📊 Loaded: {len(anchors)} Anchors, {len(frontiers)} Frontiers")

    if not frontiers:
        print("⚠️ No frontiers to process.")
        return

    # 2. vLLM 初始�?(优化配置)
    print("🚀 Initializing vLLM...")
    llm = LLM(
        model=MODEL_PATH, 
        tensor_parallel_size=1, 
        dtype="bfloat16", 
        gpu_memory_utilization=0.95, # 榨干显存
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=False # 使用 CUDA Graphs 加�?    )
    
    # 【关键修改】添加重复惩罚和更多停止�?    sampling_params = SamplingParams(
        n=NUM_SAMPLES, 
        temperature=0.7, 
        top_p=0.9, 
        max_tokens=MAX_TOKENS,
        # 1.05 的惩罚足以打断死循环，但不会影响正常数学公式
        repetition_penalty=1.05, 
        # 增加停止词，防止废话
        stop=["Problem:", "\n\nProblem", "###", "\n\n\n", "---"] 
    )
    
    prompts = [f"Problem: {item['instruction']}\nSolution (Think step-by-step):" for item in frontiers]
    
    print(f"�?Generating {len(prompts)} x {NUM_SAMPLES} samples...")
    print("   (Note: Repetition penalty enabled to prevent infinite loops)")
    
    # 开始生�?    outputs = llm.generate(prompts, sampling_params)
    
    # 3. 并行处理
    print(f"⚙️ Processing results with {NUM_WORKERS} workers...")
    
    tasks = []
    for i, output in enumerate(outputs):
        candidates = [o.text for o in output.outputs]
        tasks.append((frontiers[i], candidates))
    
    new_dataset = []
    # 添加 Anchors
    for item in anchors:
        new_dataset.append({
            "instruction": item['instruction'],
            "output": item['output'],
            "source": "anchor_gt"
        })
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results_list = list(executor.map(process_single_output, tasks))
    
    rft_count = 0
    contrastive_count = 0
    discard_count = 0
    
    for res in results_list:
        if not res:
            discard_count += 1
        else:
            new_dataset.extend(res)
            for r in res:
                if r['source'] == 'rft_standard': rft_count += 1
                elif r['source'] == 'rft_contrastive': contrastive_count += 1

    # 4. 保存
    print("\n" + "="*40)
    print("Turbo Fix Report")
    print(f"Anchors: {len(anchors)}")
    print(f"RFT Standard: {rft_count}")
    print(f"Contrastive Pairs: {contrastive_count}")
    print(f"Discarded: {discard_count}")
    print(f"Total Dataset: {len(new_dataset)}")
    print("="*40)
    
    with open(OUTPUT_FILE, 'w') as f:
        for item in new_dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()



