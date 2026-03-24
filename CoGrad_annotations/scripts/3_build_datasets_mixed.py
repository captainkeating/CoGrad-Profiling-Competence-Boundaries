#!/usr/bin/env python3
# stage3_construct_math_dataset_v2.py
# 核心功能: 根据SliCK标注结果，构建用于模型微调的、具有不同知识配比的实验数据集�?# 主要功能分为两部分：
# 1. 构建全局比例变体: 混合 "Never Correct" 数据和所�?"Known" (Highly+Maybe+Weakly) 数据�?#    生成不同 "Never Correct" 知识占比 (�?%, 10%, ..., 100%) 的数据集，用于研究新知识注入的影响�?# 2. 构建特定类别配对混合: 允许指定两个SliCK类别 (�?Intermittently Correct vs Consistently Correct)�?#    并生成这两个类别按不同比例混合的数据集，用于更精细地研究不同难度知识之间的影响�?
import json
import os
import random
import shutil
from collections import defaultdict
from typing import List, Dict

class MathDatasetConstructor:
    def __init__(self, input_file: str, output_base_dir: str):
        self.input_file = input_file
        self.output_base_dir = output_base_dir
        
        # 论文定义的知识类�?        self.categories = ["Consistently Correct", "Intermittently Correct", "Marginally Correct", "Never Correct"]
        
        # 随机种子保证可复现�?        random.seed(42)
        
        # 数据�?        self.data_pool = defaultdict(list)
        
    def load_data(self):
        """加载 Stage 2 生成的带标签数据"""
        print(f"正在加载 SliCK 标注数据: {self.input_file}")
        
        valid_count = 0
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    label = item.get('CoGrad_label')
                    
                    if label in self.categories:
                        self.data_pool[label].append(item)
                        valid_count += 1
                except:
                    continue
                    
        print(f"\n=== 数据分布统计 ===")
        for cat in self.categories:
            print(f"  {cat:<12}: {len(self.data_pool[cat])} �?)
        print(f"  总有效数�?: {valid_count} �?)

    def determine_global_D(self):
        """
        [原逻辑] 计算全局数据集大\uff1fD (用于 Never Correct vs All Known)
        """
        n_never_correct = len(self.data_pool["Never Correct"])
        
        # Known Pool = Highly + Maybe + Weakly
        self.known_pool_items = (
            self.data_pool["Consistently Correct"] + 
            self.data_pool["Intermittently Correct"] + 
            self.data_pool["Marginally Correct"]
        )
        n_known = len(self.known_pool_items)
        
        self.global_D = min(n_never_correct, n_known)
        
        print(f"\n=== [Group 1] Never Correct vs All Known settings ===")
        print(f"  Never Correct Pool: {n_never_correct} | Known Pool: {n_known}")
        print(f"  --> Global D = {self.global_D}")

    def build_ratio_variants(self):
        """
        [原逻辑] 构建 Never Correct 比例变体 (Target=Never Correct, Base=All Knowns)
        """
        ratios = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        print(f"\n正在构建 Group 1 (Never Correct vs All Known)...")
        
        for ratio in ratios:
            n_never_correct_target = int(self.global_D * ratio)
            n_known_target = self.global_D - n_never_correct_target
            
            selected_never_correct = random.sample(self.data_pool["Never Correct"], n_never_correct_target)
            selected_known = random.sample(self.known_pool_items, n_known_target)
            
            combined = selected_never_correct + selected_known
            random.shuffle(combined)
            
            # 命名保持原样
            folder_name = f"D_{int(ratio*100)}Pct_Never Correct"
            self._save_dataset(folder_name, combined)

    def build_pairwise_mixing(self, target_cat: str, base_cat: str):
        """
        [新功能] 构建两个特定类别的混合数据集
        :param target_cat: 目标类别 (例如 Intermittently Correct)，其比例会从 0% 变到 100%
        :param base_cat:   基础类别 (例如 Consistently Correct)，用于填充剩余部�?        """
        print(f"\n=== [Group Mix] {target_cat} (Target) vs {base_cat} (Base) ===")
        
        pool_target = self.data_pool[target_cat]
        pool_base = self.data_pool[base_cat]
        
        count_target = len(pool_target)
        count_base = len(pool_base)
        
        # 计算该特定组合的瓶颈 D_local
        # 我们希望在该组对比实验中，总数据量保持一�?        D_local = min(count_target, count_base)
        
        print(f"  {target_cat}: {count_target} | {base_cat}: {count_base}")
        print(f"  --> Local D ({target_cat}_{base_cat}) = {D_local}")
        
        if D_local < 50:
            print(f"⚠️ 警告: 数据�?({D_local}) 过小，跳过该组构建�?)
            return

        ratios = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        
        for ratio in ratios:
            # 计算数量
            n_target = int(D_local * ratio)
            n_base = D_local - n_target
            
            # 随机采样
            selected_target = random.sample(pool_target, n_target)
            selected_base = random.sample(pool_base, n_base)
            
            # 合并打乱
            combined = selected_target + selected_base
            random.shuffle(combined)
            
            # 文件夹命�? D_Base_Target_50Pct
            # 例如: D_Highly_Maybe_50Pct (代表 Maybe �?50%, Highly �?50%)
            # 为了文件名简短，我们截取类别的前几个字母或完整单�?            
            # 命名格式：D_{Base类别}_{Target类别}_{Target比例}Pct
            # 这种命名方式方便在训练脚本中识别是谁混进了谁
            base_short = base_cat.replace("Known", "") # Highly
            target_short = target_cat.replace("Known", "") # Maybe
            
            folder_name = f"D_{base_short}_{target_short}_{int(ratio*100)}Pct"
            
            self._save_dataset(folder_name, combined)

    def _save_dataset(self, folder_name: str, items: List[Dict]):
        """保存�?jsonl"""
        output_dir = os.path.join(self.output_base_dir, folder_name)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, "train.jsonl")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in items:
                # 兼容不同格式�?key
                if 'problem' in item: q = item['problem']
                elif 'question' in item and isinstance(item['question'], dict): q = item['question'].get('problem', "")
                else: q = str(item.get('question', ""))
                
                if 'solution' in item: a = item['solution']
                elif 'answer' in item: a = item['answer']
                elif 'question' in item and isinstance(item['question'], dict): a = item['question'].get('ground_truth_answer', "")
                else: a = ""
                
                finetune_item = {
                    "instruction": q,
                    "input": "", 
                    "output": a,
                    "CoGrad_label": item.get('CoGrad_label'),
                    "mix_group": folder_name # 方便追踪来源
                }
                
                f.write(json.dumps(finetune_item, ensure_ascii=False) + '\n')
        
        print(f"  已生�? {folder_name:<30} | 样本�? {len(items)}")

def main():
    # === 配置路径 ===
    INPUT_FILE = "/data/zhang_youhui/DoesFineTuning/DoesFineTuning_math500/stage1_labeled/train_labeled.jsonl"
    OUTPUT_DIR = "/data/zhang_youhui/DoesFineTuning/DoesFineTuning_math500/finetuning_datasets"
    
    if not os.path.exists(INPUT_FILE):
        print(f"�?错误: 找不到输入文�?{INPUT_FILE}")
        return

    constructor = MathDatasetConstructor(INPUT_FILE, OUTPUT_DIR)
    
    # 1. 加载数据
    constructor.load_data()
    
    # 2. Group A: 原始论文逻辑 (Never Correct vs All Knowns)
    constructor.determine_global_D()
    constructor.build_ratio_variants()
    
    # 3. Group B: Intermittently Correct (Target) vs Consistently Correct (Base)
    # 文件夹示�? D_Highly_Maybe_50Pct (Maybe �?50%)
    constructor.build_pairwise_mixing(target_cat="Intermittently Correct", base_cat="Consistently Correct")
    
    # 4. Group C: Marginally Correct (Target) vs Consistently Correct (Base)
    # 文件夹示�? D_Highly_Weakly_50Pct (Weakly �?50%)
    constructor.build_pairwise_mixing(target_cat="Marginally Correct", base_cat="Consistently Correct")
    
    print(f"\n�?所有微调数据集构建完成！路�? {OUTPUT_DIR}")

if __name__ == "__main__":
    main()



