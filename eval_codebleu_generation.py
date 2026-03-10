import os
import json
import re
import torch
import math
import argparse
import sys
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================================================
# 1. 挂载本地 CodeBLEU 路径
# =========================================================
# 请确保此路径正确指向包含 CodeBLEU 文件夹的父目录
LOCAL_EVAL_PATH = "/data/private/ExeCoder/evaluation/evaluator/match_based"
sys.path.append(LOCAL_EVAL_PATH)

from CodeBLEU import calc_code_bleu
from CodeBLEU.parser.DFG import DFG_python, DFG_java, DFG_csharp
from tree_sitter import Language


# =========================================================
# 3. DFG 映射配置 (含 Dummy)
# =========================================================
def DFG_dummy(root_node, index_to_code, states):
    """用于 C++ 的空 DFG 函数，返回空列表，避免报错"""
    return []

code2DFG = {
    'java': DFG_java,
    'python': DFG_python,
    'c_sharp': DFG_csharp,
    'c#': DFG_csharp,
    'cpp': DFG_dummy,  # C++ 使用空函数
    'c++': DFG_dummy,
}

# ================= 配置区域 =================
# 请根据实际情况修改
MODEL_PATH = "/data/private/ExeCoder/cg_results/Deepseek-coder-6.7b-1epoch-code" 
DATA_PATH = "/data/private/ExeCoder/data/XLCoST_data/XLCoST-Instruct/Tuning/code/train.json"
OUTPUT_BASE_DIR = os.path.join(MODEL_PATH, "codebleu_results")

DEFAULT_SYSTEM_PROMPT = "You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions."

# PROMPT_DICT = {
#     "prompt_input": (
#         "<|im_start|>system\n"
#         "{system_prompt}<|im_end|>\n"
#         "<|im_start|>user\n"
#         "{instruction}\n\n"
#         "{input}<|im_end|>\n"
#         "<|im_start|>assistant\n<think>\n</think>\n"
#     ),
#     "prompt_no_input": (
#         "<|im_start|>system\n"
#         "{system_prompt}<|im_end|>\n"
#         "<|im_start|>user\n"
#         "{instruction}<|im_end|>\n"
#         "<|im_start|>assistant\n<think>\n</think>\n"
#     ),
# }

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n"
        "{instruction}\n"
        "{input}\n"
        "### Response:"
    ),
    
    "prompt_no_input": (
        "### Instruction:\n"
        "{instruction}\n"
        "### Response:"
    ),
}

# PROMPT_DICT = {
#     # 场景 A: 包含 instruction (题目) 和 input (具体输入/上下文)
#     "prompt_input": (
#         "@@ Instruction\n"
#         "{instruction}\n"
#         "{input}\n\n"
#         "@@ Response"
#     ),
    
#     # 场景 B: 只包含 instruction (题目)
#     "prompt_no_input": (
#         "@@ Instruction\n"
#         "{instruction}\n"
#         "@@ Response"
#     ),
# }

# ================= 辅助函数 =================

def extract_code_content(text):
    """
    鲁棒的代码提取：去除 Markdown 标记
    增强正则：支持 c++, objective-c 等带符号的语言名
    """
    if not text: return ""
    # [\w\+\-]+ 允许匹配 cpp, c++, my-lang 等
    pattern = r"```(?:[\w\+\-]+)?\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def format_prompt(instruction, inp):
    if inp and inp.strip() != "":
        return PROMPT_DICT["prompt_input"].format(
            instruction=instruction, input=inp, system_prompt=DEFAULT_SYSTEM_PROMPT
        )
    else:
        return PROMPT_DICT["prompt_no_input"].format(
            instruction=instruction, system_prompt=DEFAULT_SYSTEM_PROMPT
        )

def parse_category_and_lang(instruction):
    """解析转换方向，并返回标准化的语言 Key"""
    instruction_lower = instruction.lower()
    
    # CodeBLEU 识别的 Key
    lang_map = {
        "c++": "cpp", "cpp": "cpp", 
        "python": "python", 
        "java": "java",
        "c#": "c_sharp", "csharp": "c_sharp",
        "go": "go", "php": "php", "javascript": "javascript"
    }
    
    # 简单的正则匹配 from X to Y
    match = re.search(r"from\s+(.*?)\s+to\s+(.*?)(?:\s|$|\.)", instruction_lower)
    if match:
        src_raw = match.group(1).strip()
        tgt_raw = match.group(2).strip().rstrip('.') # 去除句号
        
        src_std = lang_map.get(src_raw, src_raw)
        tgt_std = lang_map.get(tgt_raw, tgt_raw)
        
        return f"{src_std}2{tgt_std}", tgt_std
    
    return "unknown", "python"

def compute_local_codebleu(gt_code, rollout_code, lang):
    """
    调用本地 CodeBLEU 计算分数
    """
    clean_gt = extract_code_content(gt_code)
    clean_rollout = extract_code_content(rollout_code)

    if not clean_gt.strip() or not clean_rollout.strip():
        return 0.0

    # 获取 DFG 函数 (C++ 会拿到 Dummy)
    dfg_func = code2DFG.get(lang, DFG_dummy)
    
    # 针对 C++ 调整权重策略 (因为 DataFlow 是废的)
    # 尝试把权重分配给前三项: 0.25 -> 0.33
    if lang == 'cpp':
        weights = '0.3333,0.3333,0.3333,0.0'
    else:
        weights = '0.25,0.25,0.25,0.25'

    try:
        # 尝试调用带 params 参数的版本
        # 输入维度修复：[[gt]], [rollout]
        score = calc_code_bleu.get_codebleu_list(
            [[clean_gt]], 
            [clean_rollout], 
            lang,
            params=weights
        )
        return score

    except TypeError:
        # 如果本地版本不支持 params 参数，回退到默认调用
        # 此时 C++ 分数满分可能只有 0.75，但依然有效
        try:
            score = calc_code_bleu.get_codebleu_list(
                [[clean_gt]], 
                [clean_rollout], 
                lang
            )
            return score
        except:
            return 0.0
            
    except Exception as e:
        # 仅在调试时取消注释，防止刷屏
        # print(f"Error ({lang}): {e}")
        return 0.0

def calculate_loss(model, tokenizer, prompt_str, ground_truth_str, device):
    full_text = prompt_str + ground_truth_str + tokenizer.eos_token
    input_ids = tokenizer(full_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    prompt_ids = tokenizer(prompt_str, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    
    prompt_len = prompt_ids.shape[1]
    labels = input_ids.clone()
    mask_len = min(prompt_len, labels.shape[1])
    labels[:, :mask_len] = -100
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    return outputs.loss.item()

def generate_rollout(model, tokenizer, prompt_str, device):
    inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs.input_ids.shape[1]
    output_ids = generated_ids[0][input_len:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)

# ================= 主流程 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(f"cuda:0")
    print(f"[Rank {args.rank}] Loading model on {device}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 加载数据
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        try:
            full_data = json.load(f)
        except:
            f.seek(0)
            full_data = [json.loads(line) for line in f]

    total_len = len(full_data)
    chunk_size = math.ceil(total_len / args.world_size)
    start_idx = args.rank * chunk_size
    end_idx = min(start_idx + chunk_size, total_len)
    
    local_data = full_data[start_idx:end_idx]
    print(f"[Rank {args.rank}] Processing {len(local_data)} samples...")

    local_results = []

    for idx, item in tqdm(enumerate(local_data), total=len(local_data), position=args.rank):
        real_idx = start_idx + idx
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        gt_output = str(item.get("output", ""))
        
        category, target_lang = parse_category_and_lang(instruction)
        prompt = format_prompt(instruction, inp)
        
        # 1. Loss
        try:
            loss = calculate_loss(model, tokenizer, prompt, gt_output, device)
        except:
            loss = 999.0

        # 2. Rollout
        rollout = generate_rollout(model, tokenizer, prompt, device)
        
        # 3. CodeBLEU
        cb_score = compute_local_codebleu(gt_output, rollout, target_lang)
        
        sample_id = f"{category}_{real_idx}"
        
        local_results.append({
            "category": category,
            "data": {
                "id": sample_id,
                "loss": loss,
                "p_score": 1 - cb_score + 0.001,
                "codebleu": cb_score,
                "gt_code": gt_output,
                "rollout": rollout,
                "instruction": instruction,
                "input": inp
            }
        })
        
        # 保存 TXT (供人工检查)
        cat_dir = os.path.join(OUTPUT_BASE_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)
        try:
            with open(os.path.join(cat_dir, f"{sample_id}.txt"), 'w', encoding='utf-8') as f:
                f.write(f"ID: {sample_id}\nCodeBLEU: {cb_score:.5f}\nLoss: {loss:.5f}\n")
                f.write("-" * 30 + "\nGT:\n" + gt_output + "\n\nRollout:\n" + rollout)
        except: pass

    # 保存 JSONL
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_BASE_DIR, f"temp_rank_{args.rank}.jsonl"), 'w', encoding='utf-8') as f:
        for item in local_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Rank {args.rank}] Done!")

if __name__ == "__main__":
    main()