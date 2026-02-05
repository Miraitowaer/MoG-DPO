import os
import re
import json
import torch
import logging
import tempfile
import subprocess
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional, List, Any
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainerCallback,
    HfArgumentParser
)
from trl import GRPOTrainer, GRPOConfig

# ================= 配置与工具 =================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT_DICT = {
    "prompt_input": (
        "<|im_start|>system\n"
        "{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        "{instruction}\n\n"
        "{input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "prompt_no_input": (
        "<|im_start|>system\n"
        "{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        "{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}
DEFAULT_SYSTEM_PROMPT = "You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions."

@dataclass
class ScriptArguments:
    sft_model_path: str = field(metadata={"help": "SFT 模型路径"})
    data_path: str = field(metadata={"help": "训练数据路径 (JSON/JSONL)"})
    max_length: int = field(default=2048, metadata={"help": "最大序列长度"})

# ================= 模块 A: 执行器 (并行优化版) =================
class CodeExecutor:
    @staticmethod
    def detect_language(instruction: str, gt_code: str) -> str:
        if "```cpp" in gt_code or "```c++" in gt_code: return "cpp"
        if "```python" in gt_code: return "python"
        if "```java" in gt_code: return "java"
        instruction = instruction.lower()
        if " to cpp" in instruction or " to c++" in instruction: return "cpp"
        if " to java" in instruction: return "java"
        if " to python" in instruction: return "python"
        return "cpp" 

    @staticmethod
    def run_code_sync(code: str, lang: str, input_str: str = "") -> tuple[bool, str]:
        # 清理
        code = re.sub(r'```[a-zA-Z]*', '', code).replace('```', '').strip()
        is_success = False
        output_str = ""
        try:
            if lang in ['python', 'py']:
                with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as f:
                    f.write(code)
                    file_path = f.name
                # 3秒超时
                proc = subprocess.run(['python3', file_path], input=input_str, text=True, capture_output=True, timeout=3)
                if proc.returncode == 0:
                    is_success = True; output_str = proc.stdout.strip()
                os.remove(file_path)
            elif lang in ['cpp', 'c++']:
                with tempfile.NamedTemporaryFile(suffix=".cpp", mode='w', delete=False) as f:
                    f.write(code)
                    src_path = f.name
                bin_path = src_path + ".out"
                # O2 优化
                comp = subprocess.run(['g++', '-O2', '-w', src_path, '-o', bin_path], capture_output=True, text=True)
                if comp.returncode == 0:
                    try:
                        proc = subprocess.run([bin_path], input=input_str, text=True, capture_output=True, timeout=2)
                        if proc.returncode == 0: is_success = True; output_str = proc.stdout.strip()
                    except subprocess.TimeoutExpired: pass
                    if os.path.exists(bin_path): os.remove(bin_path)
                if os.path.exists(src_path): os.remove(src_path)
            elif lang == 'java':
                class_name_match = re.search(r'class\s+(\w+)', code)
                class_name = class_name_match.group(1) if class_name_match else "Main"
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(temp_dir, f"{class_name}.java")
                    with open(file_path, 'w') as f: f.write(code)
                    comp = subprocess.run(['javac', file_path], capture_output=True, text=True)
                    if comp.returncode == 0:
                        try:
                            proc = subprocess.run(['java', '-cp', temp_dir, class_name], input=input_str, text=True, capture_output=True, timeout=3)
                            if proc.returncode == 0: is_success = True; output_str = proc.stdout.strip()
                        except subprocess.TimeoutExpired: pass
        except Exception: pass 
        return is_success, output_str

# ================= 模块 B: Reward Manager (修复串行瓶颈) =================
class RewardManager:
    def __init__(self, output_dir, num_generations, accelerator_process_index):
        self.output_dir = output_dir
        self.num_generations = num_generations
        self.rank = accelerator_process_index
        
        self.current_step = 0
        self.current_epoch = 0.0
        
        if self.rank == 0 or True: 
            self.rollout_dir = os.path.join(output_dir, "rollouts")
            os.makedirs(self.rollout_dir, exist_ok=True)

    def correctness_reward_func(self, prompts, completions, gt_code, instruction, input, **kwargs) -> List[float]:
        rewards = []
        group_size = self.num_generations
        
        # 遍历每个 Group (对应一个 Prompt)
        for i in range(0, len(completions), group_size):
            group_completions = completions[i : i + group_size]
            group_gt = gt_code[i]
            group_instr = instruction[i]
            group_inp = input[i]
            
            lang = CodeExecutor.detect_language(group_instr, group_gt)
            inp_str = group_inp if group_inp else ""
            
            # 1. 跑 GT (缓存结果会更好，这里简化处理)
            gt_success, gt_out = CodeExecutor.run_code_sync(group_gt, lang, inp_str)
            gt_out_clean = gt_out.strip()
            
            # 2. [FIX] 并行跑生成的代码，解决性能瓶颈
            cleaned_codes = [c.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip() for c in group_completions]
            
            with ThreadPoolExecutor(max_workers=min(len(cleaned_codes), 8)) as executor:
                # 提交任务
                futures = [executor.submit(CodeExecutor.run_code_sync, code, lang, inp_str) for code in cleaned_codes]
                # 获取结果
                results = [f.result() for f in futures]
            
            # 3. 评分与分类
            group_rewards = []
            perfects, syncs, faileds = [], [], []
            
            for code_clean, (gen_success, gen_out) in zip(cleaned_codes, results):
                gen_out_clean = gen_out.strip()
                score = 0.1 # Default Failed
                
                if not gt_success:
                    score = 0.0 # GT 挂了
                    faileds.append(code_clean)
                elif gen_success:
                    if gen_out_clean == gt_out_clean:
                        score = 1.0 # Perfect
                        perfects.append(code_clean)
                    else:
                        score = 0.5 # Sync
                        syncs.append(code_clean)
                else:
                    score = 0.1 # Failed
                    faileds.append(code_clean)
                
                group_rewards.append(score)
            
            rewards.extend(group_rewards)
            
            # 4. 记录日志
            debug_item = {
                "global_step": self.current_step,
                "epoch": float(self.current_epoch),
                "instruction": group_instr,
                "input": group_inp,
                "target_lang": lang,
                "gt_code": group_gt,
                "candidates": cleaned_codes,
                "rewards": group_rewards,
                "perfects_count": len(perfects),
                "syncs_count": len(syncs),
                "faileds_count": len(faileds),
                "perfects_samples": perfects[:1], 
                "syncs_samples": syncs[:3],    
                "faileds_samples": faileds[:3]
            }
            
            log_file = os.path.join(self.rollout_dir, f"epoch-{int(self.current_epoch)}_rank{self.rank}.jsonl")
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(debug_item, ensure_ascii=False) + "\n")
            except Exception: pass

        return rewards

    def format_reward_func(self, completions, **kwargs) -> List[float]:
        return [0.1 if "```" in c else 0.0 for c in completions]

# ================= 模块 C: 状态同步 Callback =================
class StepTrackerCallback(TrainerCallback):
    def __init__(self, reward_manager):
        self.rm = reward_manager
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.rm.current_step = state.global_step
        self.rm.current_epoch = state.epoch if state.epoch else 0.0

# ================= 主程序 =================
def main():
    parser = HfArgumentParser((ScriptArguments, GRPOConfig))
    script_args, grpo_args = parser.parse_args_into_dataclasses()

    # [FIX] 显式设置 DDP 超时，防止代码跑太久导致 NCCL 崩溃
    grpo_args.ddp_timeout = 10800 # 3小时

    # 1. 数据预处理
    def process_data(examples):
        prompts = []
        for instr, inp in zip(examples['instruction'], examples['input']):
            if inp: 
                prompt = PROMPT_DICT["prompt_input"].format(instruction=instr, input=inp, system_prompt=DEFAULT_SYSTEM_PROMPT)
            else: 
                prompt = PROMPT_DICT["prompt_no_input"].format(instruction=instr, system_prompt=DEFAULT_SYSTEM_PROMPT)
            prompts.append(prompt)
        return {"prompt": prompts}

    dataset = load_dataset("json", data_files=script_args.data_path, split="train")
    if "data" in dataset.column_names:
        dataset = dataset.flatten()
        dataset = dataset.rename_columns({
            "data.instruction": "instruction",
            "data.input": "input",
            "data.gt_code": "gt_code"
        })
    dataset = dataset.map(process_data, batched=True)

    # 2. 模型加载
    model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # attn_implementation="sdpa"
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 3. 初始化 Reward Manager
    rank = int(os.environ.get("RANK", 0))
    reward_manager = RewardManager(
        output_dir=grpo_args.output_dir,
        num_generations=grpo_args.num_generations,
        accelerator_process_index=rank
    )

    # 4. 初始化 Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            reward_manager.correctness_reward_func, 
            reward_manager.format_reward_func
        ],
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[StepTrackerCallback(reward_manager)] 
    )

    logger.info("Starting GRPO training with Parallel Evaluation...")
    trainer.train()
    
    trainer.save_model(grpo_args.output_dir)
    tokenizer.save_pretrained(grpo_args.output_dir)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()