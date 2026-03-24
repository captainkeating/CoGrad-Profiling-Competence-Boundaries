#!/usr/bin/env python3
# stage7_1_memory_handoff.py
# 核心功能: 实现针对大语言模型（如13B参数级别）的两阶段课程学习微调�?# 关键特性与优化�?# 1. 两阶段训�?(Two-Phase Training): 
#    - Phase 1 (Anchors Consolidation): 首先在简单、高质量的数�?(Anchors) 上进行训练，巩固模型的基础能力�?#    - Phase 2 (Frontiers + Replay): 接着在更难、更前沿的数�?(Frontiers) 上训练，同时混合一部分简单数据进行“回放�?Replay)，防止遗忘�?# 2. 内存直接交接 (Memory Handoff): Phase 1 训练完成后，不将模型、优化器等状态写入磁盘，
#    而是在内存中直接销毁Phase 1的优化器状态（通常占用巨大显存），然后将训练好的模型权重直接传递给Phase 2的Trainer�?#    这极大地降低了磁盘I/O开销，并从根本上避免了因中间产物过大导致的存储或显存溢出问题�?# 3. Safetensors 安全保存: 在Phase 2训练过程中，使用 `safetensors` 格式进行模型权重的手动、定点保存�?#    这种格式支持零拷贝（zero-copy），保存速度快，内存占用极小，进一步提升了训练过程的稳定性�?
import os
import sys
import json
import torch
import re
import argparse
import random
import math
import gc
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    set_seed
)

# ================= 🔧 参数解析 =================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="D_CoGrad_Contrastive_Only")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=8) 
parser.add_argument("--lr", type=float, default=1e-5) 
args, unknown = parser.parse_known_args()

# ================= ⚙️ 基础配置 =================
set_seed(42)
CONDA_CUDA_HOME = "/data/zhang_youhui/miniconda3/envs/Math"

os.environ["CUDA_HOME"] = CONDA_CUDA_HOME
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ⚠️ 确认路径：是 Llama 13B
MODEL_PATH = "/data/zhang_youhui/llama3-13B"
BASE_PATH = "/data/zhang_youhui/DoesFineTuning/DoesFineTuning_math500"
DATA_DIR = f"{BASE_PATH}/finetuning_datasets/{args.dataset_name}"
OUTPUT_DIR = f"{BASE_PATH}/results/13B_LLAMA/{args.dataset_name}_Full_Curriculum_Final_2"

MAX_SEQ_LENGTH = 1024
GRAD_ACCUMULATION = 4 

# ================= 🛠�?数据处理工具 =================
def clean_math_text(text):
    if not text: return ""
    text = re.sub(r'\[asy\].*?\[/asy\]', '', text, flags=re.DOTALL)
    text = " ".join(text.split())
    return text

class MathCurriculumDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, split_type="all"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        anchors = []
        frontiers = []
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        source = item.get('source', '')
                        q = clean_math_text(item.get('instruction', ''))
                        a = clean_math_text(item.get('output', ''))
                        if not q or not a: continue
                        processed_item = (q, a)
                        if 'anchor' in source or 'Consistently' in source:
                            anchors.append(processed_item)
                        else:
                            frontiers.append(processed_item)
                    except: pass
        
        if split_type == "anchors_only":
            self.data = anchors
        elif split_type == "frontiers_with_replay":
            if len(anchors) > 0:
                target_anchor_count = len(frontiers) // 2 
                repeat_factor = math.ceil(target_anchor_count / len(anchors))
                upsampled_anchors = (anchors * repeat_factor)[:target_anchor_count]
                self.data = frontiers + upsampled_anchors
                random.shuffle(self.data)
            else:
                self.data = frontiers
        else:
            self.data = anchors + frontiers
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, a = self.data[idx]
        prompt = f"Problem: {q}\nSolution (Do not use [asy] code. Think step-by-step and put the final answer in \\boxed{{}}):"
        completion = f" {a}"
        full_text = prompt + completion + self.tokenizer.eos_token
        tokens = self.tokenizer(full_text, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        input_ids = tokens.input_ids[0]
        attention_mask = tokens.attention_mask[0]
        labels = input_ids.clone()
        prompt_tokens = self.tokenizer(prompt, max_length=self.max_length, truncation=True, padding=False)
        labels[:len(prompt_tokens.input_ids)] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "length": len(input_ids)}

# ================= 🎯 【核心修复】手动保�?Callback (使用 Safetensors) =================
class SafeManualSaveCallback(TrainerCallback):
    """
    使用 safe_serialization=True 进行零拷贝保存，防止 Phase 2 存盘�?    """
    def __init__(self, target_steps, output_dir, model, tokenizer):
        self.target_steps = set(target_steps)
        self.output_dir = output_dir
        self.model = model
        self.tokenizer = tokenizer

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.target_steps:
            print(f"\n🧹 [Memory] Cleaning up before manual save at step {state.global_step}...")
            gc.collect()
            torch.cuda.empty_cache()
            
            ckpt_path = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(ckpt_path, exist_ok=True)
            
            print(f"📸 [SafeSave] Saving WEIGHTS ONLY (Safetensors) to {ckpt_path}...")
            
            # �?启用 safe_serialization=True (关键优化，内存占用极�?
            try:
                self.model.save_pretrained(ckpt_path, safe_serialization=True)
                self.tokenizer.save_pretrained(ckpt_path)
                print(f"�?Save Complete!")
            except Exception as e:
                print(f"�?Save Failed: {e}")
                print("⚠️ Continuing training anyway...") # 即使保存失败也不要让训练崩溃
            
            control.should_save = False

# ================= 🏁 主流�?=================
def main():
    print(f"\n🚀 Turbo Teacher [Memory Handoff] - No Phase 1 Save")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print(">>> Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" 
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # ---------------- Phase 1 ----------------
    print("\n🔥 Phase 1: Anchors Consolidation")
    dataset_phase1 = MathCurriculumDataset(f"{DATA_DIR}/train.jsonl", tokenizer, MAX_SEQ_LENGTH, split_type="anchors_only")
    
    args_phase1 = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/phase1_temp",
        num_train_epochs=2, 
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=args.lr,
        logging_steps=5,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        group_by_length=True,
        save_strategy="no", 
        remove_unused_columns=False,
        dataloader_num_workers=0 # 保持 0
    )

    def collate_fn_wrapper(features):
        for f in features:
            if "length" in f: del f["length"]
        return DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)(features)

    trainer = Trainer(
        model=model,
        args=args_phase1,
        train_dataset=dataset_phase1,
        tokenizer=tokenizer,
        data_collator=collate_fn_wrapper
    )
    
    if len(dataset_phase1) > 0:
        trainer.train()
        
        # ⚡⚡�?核心修改：绝对不存盘！⚡⚡⚡
        print(f"\n�?SKIPPING Phase 1 Disk Save to prevent OOM.")
        print("🧹 Cleaning up Phase 1 Optimizer memory...")
        
        # 销�?Phase 1 Trainer，释�?50GB 优化器内�?        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        print("�?Memory Cleaned! Handoff model to Phase 2 in-memory...")

    # ---------------- Phase 2 ----------------
    # 确保模型状态正�?    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    print("\n🔥 Phase 2: Frontiers + Weighted Replay")
    dataset_phase2 = MathCurriculumDataset(f"{DATA_DIR}/train.jsonl", tokenizer, MAX_SEQ_LENGTH, split_type="frontiers_with_replay")
    
    if len(dataset_phase2) > 0:
        total_samples = len(dataset_phase2)
        effective_batch_size = args.batch_size * GRAD_ACCUMULATION
        steps_per_epoch = total_samples // effective_batch_size
        
        target_step_03 = max(1, int(steps_per_epoch * 0.3))
        target_step_05 = max(2, int(steps_per_epoch * 0.5))
        target_step_epoch_2_end = max(3, steps_per_epoch * 2)
        target_step_epoch_3_end = max(4, steps_per_epoch * 3)
        target_step_epoch_5_end = max(5, steps_per_epoch * 5)

        targets = [target_step_03, target_step_05, target_step_epoch_2_end, target_step_epoch_3_end, target_step_epoch_5_end]
        
        print(f"📊 Save Points: {targets}")

        args_phase2 = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=5, 
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=GRAD_ACCUMULATION,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=5,
            bf16=True,
            report_to="none",
            group_by_length=True,
            optim="paged_adamw_8bit",
            remove_unused_columns=False,
            dataloader_num_workers=0, 
            eval_strategy="no", 
            save_strategy="no",       
            load_best_model_at_end=False
        )

        # 实例�?Safetensors 回调
        safe_save_callback = SafeManualSaveCallback(
            target_steps=targets,
            output_dir=OUTPUT_DIR,
            model=model,
            tokenizer=tokenizer
        )

        # 重新初始�?Trainer (使用同一�?model 对象)
        trainer_p2 = Trainer(
            model=model, # 这里传入的是 Phase 1 练完的那个模型对�?            args=args_phase2,
            train_dataset=dataset_phase2,
            tokenizer=tokenizer,
            data_collator=collate_fn_wrapper,
            callbacks=[safe_save_callback]
        )

        trainer_p2.train()
    
    print(f"\n�?All Done! Checkpoints should be in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()



