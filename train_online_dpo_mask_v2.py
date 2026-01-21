import os
import torch
import torch.nn.functional as F
import difflib
import multiprocessing
import subprocess
import tempfile
import re
import logging
import time
import gc
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DummyOptim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    set_seed,
    HfArgumentParser
)
from tqdm import tqdm

# ================= 全局配置 =================
IGNORE_INDEX = -100
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ALPACA_PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
#     ),
# }

# PROMPT_DICT = {
#     "prompt_input": (
#         "{instruction}\n{input}"
#     ),
#     "prompt_no_input": (
#         "{instruction}"
#     ),
# }

# PROMPT_DICT = {
#     "prompt_input": (
#         "<|im_start|>system\n"
#         "{system_prompt}<|im_end|>\n"
#         "<|im_start|>user\n"
#         "{instruction}\n\n"
#         "{input}<|im_end|>\n"
#         "<|im_start|>assistant\n"
#     ),
#     "prompt_no_input": (
#         "<|im_start|>system\n"
#         "{system_prompt}<|im_end|>\n"
#         "<|im_start|>user\n"
#         "{instruction}<|im_end|>\n"
#         "<|im_start|>assistant\n"
#     ),
# }

PROMPT_DICT = {
        # 场景 A: 包含 instruction (题目) 和 input (具体输入/上下文)
        "prompt_input": (
            "@@ Instruction\n"
            "{instruction}\n"
            "{input}\n\n"
            "@@ Response"
        ),
        
        # 场景 B: 只包含 instruction (题目)
        "prompt_no_input": (
            "@@ Instruction\n"
            "{instruction}\n"
            "@@ Response"
        ),
    }

DEFAULT_SYSTEM_PROMPT = "You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions."

@dataclass
class ScriptArguments:
    sft_model_path: str = field(metadata={"help": "SFT 模型路径"})
    data_path: str = field(metadata={"help": "训练数据路径 (JSON)"})
    output_dir: str = field(metadata={"help": "输出目录"})
    
    num_generations: int = field(default=4)
    temperature: float = field(default=0.8)
    max_new_tokens: int = field(default=512)
    
    learning_rate: float = field(default=5e-7)
    per_device_train_batch_size: int = field(default=1) 
    gradient_accumulation_steps: int = field(default=8)
    gradient_checkpointing: bool = field(default=True)
    num_train_epochs: int = field(default=1)
    beta: float = field(default=0.1)
    max_length: int = field(default=2048)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500) 

# ================= 模块 A: 执行器 (保持不变) =================
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
    def run_code_sync(code: str, lang: str, input_str: str = "") -> Tuple[bool, str]:
        code = re.sub(r'```[a-zA-Z]*', '', code).replace('```', '').strip()
        is_success = False
        output_str = ""
        try:
            if lang in ['python', 'py']:
                with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as f:
                    f.write(code)
                    file_path = f.name
                proc = subprocess.run(['python3', file_path], input=input_str, text=True, capture_output=True, timeout=3)
                if proc.returncode == 0:
                    is_success = True; output_str = proc.stdout.strip()
                os.remove(file_path)
            elif lang in ['cpp', 'c++']:
                with tempfile.NamedTemporaryFile(suffix=".cpp", mode='w', delete=False) as f:
                    f.write(code)
                    src_path = f.name
                bin_path = src_path + ".out"
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

    def batch_evaluate(self, gt_code: str, candidates: List[str], lang: str) -> Tuple[List[str], List[str]]:
        gt_success, gt_stdout = self.run_code_sync(gt_code, lang)
        if not gt_success: return [], []
        with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
            futures = [executor.submit(self.run_code_sync, code, lang) for code in candidates]
            results = [f.result() for f in futures]
        pass_list, fail_list = [], []
        for cand, (succ, out) in zip(candidates, results):
            if succ and out.strip() == gt_stdout.strip(): pass_list.append(cand)
            else: fail_list.append(cand)
        return pass_list, fail_list

# ================= 模块 B: Mask 处理器 (保持不变) =================
class MaskProcessor:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def get_dual_diff_ranges(self, chosen_code: str, rejected_code: str):
        chosen_lines = chosen_code.splitlines(keepends=True)
        rejected_lines = rejected_code.splitlines(keepends=True)
        c_offsets, pos = [], 0
        for line in chosen_lines: c_offsets.append((pos, pos + len(line))); pos += len(line)
        r_offsets, pos = [], 0
        for line in rejected_lines: r_offsets.append((pos, pos + len(line))); pos += len(line)
        matcher = difflib.SequenceMatcher(None, chosen_lines, rejected_lines)
        c_ranges, r_ranges = [], []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal': continue 
            elif tag == 'replace':
                if i1 < len(c_offsets): c_ranges.append((c_offsets[i1][0], c_offsets[min(i2-1, len(c_offsets)-1)][1]))
                if j1 < len(r_offsets): r_ranges.append((r_offsets[j1][0], r_offsets[min(j2-1, len(r_offsets)-1)][1]))
            elif tag == 'insert':
                if j1 < len(r_offsets): r_ranges.append((r_offsets[j1][0], r_offsets[min(j2-1, len(r_offsets)-1)][1]))
            elif tag == 'delete':
                if i1 < len(c_offsets): c_ranges.append((c_offsets[i1][0], c_offsets[min(i2-1, len(c_offsets)-1)][1]))
        return c_ranges, r_ranges

    def tokenize_and_mask(self, prompt: str, chosen: str, rejected: str):
        c_focus, r_focus = self.get_dual_diff_ranges(chosen, rejected)
        c_enc = self.tokenizer(prompt + chosen, truncation=True, max_length=self.max_length, return_offsets_mapping=True, return_tensors='pt')
        r_enc = self.tokenizer(prompt + rejected, truncation=True, max_length=self.max_length, return_offsets_mapping=True, return_tensors='pt')
        prompt_len = len(prompt)
        def apply_mask(enc, focus_ranges):
            input_ids = enc.input_ids[0]
            labels = input_ids.clone(); labels[:] = IGNORE_INDEX 
            offsets = enc.offset_mapping[0]
            resp_idx = 0
            for i, (s, e) in enumerate(offsets):
                if s >= prompt_len: resp_idx = i; break
            for i in range(resp_idx, len(offsets)):
                s, e = offsets[i]
                if s == 0 and e == 0: continue
                token_s, token_e = s - prompt_len, e - prompt_len
                is_focus = False
                for (fs, fe) in focus_ranges:
                    if max(token_s, fs) < min(token_e, fe): is_focus = True; break
                if is_focus: labels[i] = input_ids[i]
            return input_ids, enc.attention_mask[0], labels
        c_ids, c_mask, c_lbl = apply_mask(c_enc, c_focus)
        r_ids, r_mask, r_lbl = apply_mask(r_enc, r_focus)
        return c_ids, c_mask, c_lbl, r_ids, r_mask, r_lbl

# ================= 模块 C: 训练逻辑 (已修正指标计算) =================
def compute_dpo_loss(model, ref_model, batch, beta=0.1):
    policy_logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
    with torch.no_grad():
        ref_logits = ref_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits

    labels = batch['labels']
    logits_p = policy_logits[:, :-1, :]
    logits_r = ref_logits[:, :-1, :]
    labels_shifted = labels[:, 1:]
    
    safe_labels = labels_shifted.clone()
    safe_labels[safe_labels == IGNORE_INDEX] = 0
    
    per_token_logps_p = torch.gather(logits_p.log_softmax(-1), 2, safe_labels.unsqueeze(2)).squeeze(2)
    per_token_logps_r = torch.gather(logits_r.log_softmax(-1), 2, safe_labels.unsqueeze(2)).squeeze(2)
    
    valid_mask = (labels_shifted != IGNORE_INDEX).float()
    
    policy_logps = (per_token_logps_p * valid_mask).sum(-1)
    ref_logps = (per_token_logps_r * valid_mask).sum(-1)
    
    chosen_policy = policy_logps[::2]
    rejected_policy = policy_logps[1::2]
    chosen_ref = ref_logps[::2]
    rejected_ref = ref_logps[1::2]
    
    # === [新增] DPO 指标计算 ===
    # 1. Rewards
    chosen_rewards = beta * (chosen_policy - chosen_ref)
    rejected_rewards = beta * (rejected_policy - rejected_ref)
    
    # 2. Accuracy & Margin
    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    margins = chosen_rewards - rejected_rewards
    
    # 3. Loss
    loss = -F.logsigmoid(margins).mean()
    
    # 构造返回字典
    metrics = {
        "rewards/chosen": chosen_rewards.mean().item(),
        "rewards/rejected": rejected_rewards.mean().item(),
        "rewards/accuracies": reward_accuracies.mean().item(),
        "rewards/margins": margins.mean().item(),
        "logps/chosen": chosen_policy.mean().item(),
        "logps/rejected": rejected_policy.mean().item()
    }
    
    return loss, metrics

def compute_model_confidence(model, tokenizer, prompt, candidate_code, device):
    """
    MO-DPO 核心组件：计算生成代码的长度归一化对数似然 (Confidence Score)
    """
    model.eval() # 切换到评估模式计算概率
    
    # 构造完整输入
    # 注意：这里假设 prompt 和 code 拼接是直接相连的。
    # 如果你的 prompt 模板末尾没有换行符但 code 需要，请在这里手动调整，例如 prompt + "\n" + candidate_code
    full_text = prompt + candidate_code
    
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # 计算 prompt 的 token 长度，我们要 mask 掉 prompt 部分的 loss
    prompt_inputs = tokenizer(prompt, add_special_tokens=False) 
    prompt_len = len(prompt_inputs["input_ids"]) 

    # 边界保护
    if prompt_len >= input_ids.shape[1]:
         prompt_len = input_ids.shape[1] // 2 

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Shift logits: 预测下一个 token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # 计算每个 token 的 loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(input_ids.size(0), -1)
    
    # 只取代码生成部分 (去掉 prompt 部分)
    # prompt_len - 1 是因为 shift 之后索引对齐发生了错位
    start_idx = max(0, prompt_len - 1)
    code_losses = token_losses[:, start_idx:]
    
    # 转换成 Log Probability (Log P = -Loss)
    code_log_probs = -code_losses
    
    if code_log_probs.shape[1] == 0:
        return -float('inf') 
        
    # 计算归一化得分：总 LogProb / Token 数量
    score = code_log_probs.sum().item() / code_log_probs.size(1)
    
    return score

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    set_seed(42)
    
    # [FIX 1] 设置 3 小时超时，防止 DeepSpeed 保存或生成太慢导致 NCCL 崩溃
    ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=180))
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        log_with="tensorboard", 
        project_dir=args.output_dir,
        kwargs_handlers=[ddp_kwargs] # 注入超时配置
    )

    if accelerator.is_main_process:
        if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
        accelerator.init_trackers(project_name="runs", config=vars(args))
        logger.info(f"Initialized Online Mask-DPO Trainer. Output: {args.output_dir}")

    # 1. 加载模型
    model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    # 显存优化配置
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False 
    
    ref_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    ref_model.eval()
    ref_model.requires_grad_(False)
    ref_model = ref_model.to(accelerator.device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    optimizer = DummyOptim(model.parameters(), lr=args.learning_rate)
    
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataloader = DataLoader(dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    executor_tool = CodeExecutor()
    masker = MaskProcessor(tokenizer, args.max_length)
    
    global_step = 0
    model.train()
    
    for epoch in range(args.num_train_epochs):
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        
        for batch_data in progress_bar:
            # === Phase 1: Mining (生成) ===
            instructions = batch_data['instruction']
            inputs = batch_data['input']
            outputs = batch_data['output']
            
            batch_input_ids, batch_labels, batch_masks = [], [], []
            mined_count = 0
            
            # 清理显存以进行生成
            torch.cuda.empty_cache() 
            
            for i, instr in enumerate(instructions):
                gt_code_raw = outputs[i]
                target_lang = CodeExecutor.detect_language(instr, gt_code_raw)
                inp_data = inputs[i]
                
                if inp_data: prompt = PROMPT_DICT["prompt_input"].format(instruction=instr, input=inp_data, system_prompt=DEFAULT_SYSTEM_PROMPT)
                else: prompt = PROMPT_DICT["prompt_no_input"].format(instruction=instr, system_prompt=DEFAULT_SYSTEM_PROMPT)
                
                with torch.no_grad():
                    inputs_tokenized = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
                    # 临时开启 Cache 加速生成
                    gen_ids = accelerator.unwrap_model(model).generate(
                        **inputs_tokenized,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        num_return_sequences=args.num_generations,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True 
                    )
                
                gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                candidates = []
                for text in gen_texts:
                    parts = text.split("### Response:")
                    c = parts[1] if len(parts) > 1 else text
                    c = re.sub(r'```[a-zA-Z]*', '', c).replace('```', '').strip()
                    candidates.append(c)
                
                gt_code = re.sub(r'```[a-zA-Z]*', '', gt_code_raw).replace('```', '').strip()
                passes, fails = executor_tool.batch_evaluate(gt_code, candidates, target_lang)
                
                if not fails: continue 
                
                # chosen = passes[0] if passes else gt_code
                # rejected = max(fails, key=lambda x: difflib.SequenceMatcher(None, chosen, x).ratio())
                # =================== MO-DPO: 分层质量评估筛选 ===================
                chosen = None
                
                # [第一层] 基础正确性筛选 (Base Correctness)
                if not passes:
                    # 如果没有通过的代码，回退到 Ground Truth (Oracle Anchor)
                    chosen = gt_code
                elif len(passes) == 1:
                    # 如果只有一个通过，直接选它
                    chosen = passes[0]
                else:
                    # [第二层] 基于置信度的优选 (Confidence-based Anchor Selection)
                    # 从通过的样本中，选择模型最“确信”的那一个
                    best_score = -float('inf')
                    best_candidate = passes[0]
                    
                    # 性能优化：如果通过的样本太多，为了不拖慢训练，只看前5个
                    candidates_to_eval = passes
                    
                    for cand_code in candidates_to_eval:
                        # 注意：prompt 变量必须是你当前上下文中 input prompt 的文本
                        score = compute_model_confidence(
                            model, 
                            tokenizer, 
                            prompt,  # 确保这里使用了当前数据的 prompt 字符串
                            cand_code, 
                            accelerator.device # 或者是 model.device
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = cand_code
                    
                    chosen = best_candidate

                # [对抗挖掘] 困难负样本构建 (Hard Negative Mining)
                # 在所有失败样本中，找到与 chosen 编辑距离最近（最像）的那个
                # ratio() 越高代表越相似，即编辑距离越小
                rejected = max(fails, key=lambda x: difflib.SequenceMatcher(None, chosen, x).ratio())
                # ==============================================================
                if chosen == rejected: continue
                
                c_ids, c_mask, c_lbl, r_ids, r_mask, r_lbl = masker.tokenize_and_mask(prompt, chosen, rejected)
                batch_input_ids.extend([c_ids, r_ids])
                batch_masks.extend([c_mask, r_mask])
                batch_labels.extend([c_lbl, r_lbl])
                mined_count += 1
            
            # === Phase 2: Training (训练) ===
            torch.cuda.empty_cache()
            gc.collect()

            # [FIX 2] DDP 同步逻辑：即使没挖到数据，也要跑 Dummy Backward
            if mined_count > 0:
                max_len = max([t.size(0) for t in batch_input_ids])
                padded_ids = [F.pad(t, (0, max_len-t.size(0)), value=tokenizer.pad_token_id) for t in batch_input_ids]
                padded_masks = [F.pad(t, (0, max_len-t.size(0)), value=0) for t in batch_masks]
                padded_labels = [F.pad(t, (0, max_len-t.size(0)), value=IGNORE_INDEX) for t in batch_labels]
                
                final_batch = {
                    'input_ids': torch.stack(padded_ids).to(accelerator.device),
                    'attention_mask': torch.stack(padded_masks).to(accelerator.device),
                    'labels': torch.stack(padded_labels).to(accelerator.device)
                }
                
                with accelerator.accumulate(model):
                    loss, metrics = compute_dpo_loss(model, ref_model, final_batch, beta=args.beta)
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                
                if global_step % args.logging_steps == 0:
                    logger.info(f"Step {global_step} | Loss: {loss.item():.4f} | Acc: {metrics['rewards/accuracies']:.2f} | Mined: {mined_count}")
                    if accelerator.is_main_process:
                        log_data = {
                            "train/loss": loss.item(),
                            "train/mined_pairs": mined_count,
                            "train/epoch": epoch,
                        }
                        log_data.update(metrics)
                        accelerator.log(log_data, step=global_step)
            else:
                # Dummy Backward 保持 NCCL 队列同步
                dummy_input = tokenizer(instructions[0], return_tensors="pt").input_ids.to(accelerator.device)
                if dummy_input.size(1) > 10: dummy_input = dummy_input[:, :10]
                
                with accelerator.accumulate(model):
                    dummy_out = model(dummy_input)
                    dummy_loss = dummy_out.logits.mean() * 0.0
                    accelerator.backward(dummy_loss)
                    optimizer.step()
                    optimizer.zero_grad()
                
                if global_step % args.logging_steps == 0:
                    logger.info(f"Step {global_step} | Loss: 0.0000 (Dummy Sync) | Mined: 0")

            global_step += 1
            
            # [FIX 3] 周期性保存 (Checkpoints) - 修复死锁
            if global_step % args.save_steps == 0:
                # 等待所有卡
                accelerator.wait_for_everyone()
                
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                
                # DeepSpeed ZeRO-2 要求所有 rank 都调用 save_state
                accelerator.save_state(save_path)
                
                # 只有主进程打印日志和保存 Tokenizer
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(save_path)
                    logger.info(f"Checkpoint saved to {save_path}")

    # ================= 训练结束，保存最终模型 =================
    
    # 1. 强制同步
    accelerator.wait_for_everyone()
    
    # 2. 解包模型 (去除 DDP/ZeRO 壳)
    unwrapped_model = accelerator.unwrap_model(model)
    
    # 3. 主进程保存为 Safetensors (用于推理)
    if accelerator.is_main_process:
        logger.info("Training finished. Saving final model to Safetensors...")
        
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=True,
            save_function=accelerator.save,
            safe_serialization=True,      # [FIX 4] 直接生成 .safetensors
            max_shard_size="10GB"
        )
        
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Final model saved to {args.output_dir}")

    # 4. 结束训练
    accelerator.end_training()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()