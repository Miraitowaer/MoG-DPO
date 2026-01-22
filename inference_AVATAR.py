import argparse
import re
import os
import sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ==========================================
# 1. 数据预处理 (AVATAR Python -> Tagged Standard Python)
# ==========================================
def preprocess_avatar_python(line: str) -> str:
    """
    将 AVATAR 数据集的一行 Python 代码还原为多行格式。
    注意：根据用户需求，保留了 Token 间的空格（如 'arr [ i ]'），
    只处理结构性 Token (NEW_LINE, INDENT, DEDENT)。
    """
    # 移除元数据标签
    try:
        line = re.sub(r'\'', '', line).strip()
    except Exception:
        pass

    tokens = line.split()
    output = []
    indent_level = 0
    
    for token in tokens:
        if token == "NEW_LINE":
            output.append("\n")
            output.append("    " * indent_level)
        elif token == "INDENT":
            indent_level += 1
            output.append("    ")
        elif token == "DEDENT":
            if indent_level > 0:
                indent_level -= 1
                # 回退缩进：尝试移除尾部的4个空格
                if output and output[-1] == "    ":
                    output.pop()
                elif output and output[-1].endswith("    "):
                    output[-1] = output[-1][:-4]
        else:
            # 普通 Token：如果不紧跟在换行符或缩进后，添加空格分隔
            if output and output[-1] != "\n" and not output[-1].endswith("    "):
                 output.append(" ")
            output.append(token)
            
    code = "".join(output)
    # 注意：这里不再执行 remove spaces around () [] 等操作，保留原始分词风格
    return code

# ==========================================
# 2. 数据后处理 (Standard Java -> AVATAR Tokenized Java)
# ==========================================
def postprocess_to_avatar_java(code: str) -> str:
    """
    提取生成的代码并转换为 AVATAR 评测所需的单行 token 化格式。
    """
    # 1. 尝试提取 <Code> 标签内的内容 (如果模型学会了输出标签)
    # 或者提取 markdown 代码块
    if "```" in code:
        match = re.search(r'```(?:java)?\s*(.*?)\s*```', code, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1)
            
    # 2. 移除注释
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # 3. 换行转空格
    code = code.replace('\n', ' ').replace('\r', ' ')
    
    # 4. 符号分词 (关键步骤，确保 CodeBLEU 计算正确)
    specials = r'([.,;(){}\[\]+\-*/%^&|!=<>?])'
    # code = re.sub(specials, r' \1 ', code)
    
    # 5. 清洗多余空格
    code = re.sub(r'\s+', ' ', code).strip()
    
    return code

# ==========================================
# 3. 模型特定配置 (Stop Tokens)
# ==========================================
def get_stop_tokens(model_path: str):
    """根据模型类型返回停止符"""
    model_name = model_path.lower()
    if "qwen" in model_name:
        return ["<|im_end|>", "<|endoftext|>"]
    elif "deepseek" in model_name:
        return ["<|EOT|>", "<|end_of_sentence|>", "</s>"]
    elif "llama" in model_name:
        return ["</s>", "<|EOT|>"]
    elif "magicoder" in model_name:
        return ["</s>"]
    return ["</s>"] # 默认

# ==========================================
# 4. 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Universal vLLM Inference Script with Custom Instruction")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--input_file", type=str, required=True, help="test.python 文件")
    parser.add_argument("--output_file", type=str, required=True, help="输出结果文件")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="GPU并行数量")
    # 默认为 16384 以防止 Context Length Error
    parser.add_argument("--max_model_len", type=int, default=16384, help="最大上下文长度")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="显存占用比例")
    args = parser.parse_args()

    # 定义固定的 Instruction
    SYSTEM_INSTRUCTION = "Translate the given code from python to java. The input Code is marked with <Code> and </Code>."

    # 加载 Tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

    # 读取数据
    print(f"Reading inputs from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    prompts = []
    print("Constructing prompts with <Code> tags...")
    
    for i, line in enumerate(raw_lines):
        # 1. 还原代码结构（保留分词空格）
        source_code = preprocess_avatar_python(line)
        
        # 2. 构建符合要求的输入格式
        # 格式：<Code>\n代码内容\n</Code>
        input_content = f"<Code>\n{source_code}\n</Code>"
        
        # 3. 构建 Chat 消息
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": input_content}
        ]
        
        # 4. 应用 Chat 模板
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            # 回退策略：如果模型没有模板（比较少见），手动拼接
            if i == 0: print(f"Warning: apply_chat_template failed ({e}), using fallback.")
            # prompt_text = f"{SYSTEM_INSTRUCTION}\n\nUser: {input_content}\n\nAssistant:"
            # CodeLlama
            prompt_text = f"[INST] <<SYS>>\n{SYSTEM_INSTRUCTION}\n<</SYS>>\n\n{input_content} [/INST]"

        prompts.append(prompt_text)
        
        # 调试：打印第一条 Prompt 确认格式
        if i == 0:
            print("\n" + "="*40)
            print(f"DEBUG: First Prompt Preview ({args.model_path}):")
            print(prompt_text)
            print("="*40 + "\n")

    # 初始化 vLLM
    print(f"Initializing vLLM Engine (max_len={args.max_model_len})...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False 
    )

    # 获取停止符
    stop_tokens = get_stop_tokens(args.model_path)
    print(f"Using stop tokens: {stop_tokens}")

    sampling_params = SamplingParams(
        temperature=0,        # 贪婪采样
        max_tokens=2048,      # 输出最大长度
        stop=stop_tokens
    )

    # 批量推理
    print(f"Generating {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)

    # 保存结果
    processed_results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        # 后处理
        final_code = postprocess_to_avatar_java(generated_text)
        processed_results.append(final_code)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for res in processed_results:
            f.write(res + "\n")

    print(f"Done. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()