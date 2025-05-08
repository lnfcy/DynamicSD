import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM
from sampling import autoregressive_sampling, speculative_sampling
from globals import Decoder

def parse_args():
    parser = argparse.ArgumentParser(description="Speculative Decoding Demo")
    parser.add_argument("--input", type=str, required=True, help="Input text for generation")
    parser.add_argument("--target_model_name", type=str, required=True, help="Target (large) model name")
    parser.add_argument("--approx_model_name", type=str, required=True, help="First approximate (large draft) model name")
    parser.add_argument("--approx_model_name2", type=str, help="Second approximate (small draft) model name")
    parser.add_argument("--max_tokens", type=int, default=1000, help="Maximum number of tokens to generate")
    parser.add_argument("--gamma", type=int, default=4, help="Number of tokens to speculate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--entropy_threshold", type=float, default=0.3, help="Entropy threshold for SVIP draft model selection")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def load_model_and_tokenizer(model_name, device):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if "cuda" in device else torch.float32)
    model = model.to(device)
    model.eval()
    return model, tokenizer

def run_generation(method_name, input_ids, generate_func, *args, **kwargs):
    print(f"\n{Fore.CYAN}Running {method_name}...{Style.RESET_ALL}")
    
    with contexttimer.Timer() as timer:
        output_ids = generate_func(*args, **kwargs)
    
    num_new_tokens = output_ids.shape[1] - input_ids.shape[1]
    elapsed_time = timer.elapsed
    tokens_per_second = num_new_tokens / elapsed_time
    
    print(f"{Fore.GREEN}Generated {num_new_tokens} tokens in {elapsed_time:.2f} seconds")
    print(f"Performance: {tokens_per_second:.2f} tokens/s{Style.RESET_ALL}")
    
    return {
        "method": method_name,
        "output_ids": output_ids,
        "tokens": num_new_tokens,
        "time": elapsed_time,
        "tokens_per_second": tokens_per_second
    }

def generate_autoregressive(input_ids, target_model, max_tokens, temp, top_k, top_p):
    """Standard autoregressive generation."""
    return autoregressive_sampling(input_ids, target_model, max_tokens, temp, top_k, top_p)

def generate_speculative_single(input_ids, target_model, approx_model, max_tokens, gamma, temp, top_k, top_p, seed=None, verbose=False):
    """Speculative decoding with a single draft model."""
    # 单模型模式不使用SVIP
    return speculative_sampling(input_ids, approx_model, target_model, max_tokens, gamma, temp, top_k, top_p, 
                                verbose=verbose, random_seed=seed, use_svip=False)

def generate_speculative_dual_svip(input_ids, target_model, approx_model1, approx_model2, max_tokens, gamma, temp, top_k, top_p, entropy_threshold=0.3, seed=None, verbose=False):
    """Speculative decoding with two draft models using SVIP entropy-based model selection."""
    # 使用SVIP策略选择模型：熵值高用大模型，熵值低用小模型
    return speculative_sampling(
        input_ids, 
        approx_model1,  # 大模型
        target_model, 
        max_tokens, 
        gamma, 
        temp, 
        top_k, 
        top_p, 
        verbose=verbose,
        approx_model2=approx_model2,  # 小模型
        random_seed=seed,
        entropy_threshold=entropy_threshold,  # SVIP熵阈值
        use_svip=True  # 启用SVIP模式
    )

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # Load tokenizer and models
    target_model, tokenizer = load_model_and_tokenizer(args.target_model_name, args.device)
    approx_model1, _ = load_model_and_tokenizer(args.approx_model_name, args.device)
    approx_model2 = None
    if args.approx_model_name2:
        approx_model2, _ = load_model_and_tokenizer(args.approx_model_name2, args.device)
    
    # Tokenize input
    input_text = args.input
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(args.device)
    
    print(f"\n{Fore.YELLOW}Input: {input_text}{Style.RESET_ALL}")
    
    # Run all generation methods and collect results
    results = []
    
    # Method 1: Autoregressive sampling with target model
    results.append(run_generation(
        "Target Model Autoregressive",
        input_ids,
        generate_autoregressive,
        input_ids, 
        target_model, 
        args.max_tokens, 
        args.temperature, 
        args.top_k, 
        args.top_p
    ))
    
    # Method 2: Speculative decoding with target + approx_model1
    results.append(run_generation(
        "Target + Large Draft Speculative",
        input_ids,
        generate_speculative_single,
        input_ids, 
        target_model, 
        approx_model1, 
        args.max_tokens, 
        args.gamma, 
        args.temperature, 
        args.top_k, 
        args.top_p,
        args.seed,
        args.verbose
    ))
    
    # Method 3: Speculative decoding with target + approx_model2 (if available)
    if approx_model2:
        results.append(run_generation(
            "Target + Small Draft Speculative",
            input_ids,
            generate_speculative_single,
            input_ids, 
            target_model, 
            approx_model2, 
            args.max_tokens, 
            args.gamma, 
            args.temperature, 
            args.top_k, 
            args.top_p,
            args.seed,
            args.verbose
        ))
    
    # Method 4: Speculative decoding with SVIP entropy-based model selection (if two models available)
    if approx_model2:
        results.append(run_generation(
            f"Target + SVIP Entropy-based Model Selection (threshold={args.entropy_threshold})",
            input_ids,
            generate_speculative_dual_svip,
            input_ids, 
            target_model, 
            approx_model1,  # 大模型，熵高时使用
            approx_model2,  # 小模型，熵低时使用
            args.max_tokens, 
            args.gamma, 
            args.temperature, 
            args.top_k, 
            args.top_p,
            args.entropy_threshold,
            args.seed,
            args.verbose
        ))
    
    # Display output for each method
    print(f"\n{Fore.MAGENTA}Generation Results:{Style.RESET_ALL}")
    print("-" * 80)
    for i, result in enumerate(results):
        output_text = tokenizer.decode(result["output_ids"][0], skip_special_tokens=True)
        print(f"\n{Fore.BLUE}Method {i+1}: {result['method']}{Style.RESET_ALL}")
        print(f"Output: {output_text}")
    
    # Summarize performance
    print(f"\n{Fore.MAGENTA}Performance Summary:{Style.RESET_ALL}")
    print("-" * 80)
    print(f"{'Method':<52} {'Tokens':<10} {'Time (s)':<10} {'Tokens/s':<10}")
    print("-" * 80)
    for result in results:
        print(f"{result['method']:<52} {result['tokens']:<10.0f} {result['time']:<10.2f} {result['tokens_per_second']:<10.2f}")
    
    if approx_model2:
        baseline_speed = results[0]["tokens_per_second"]  # 自回归基准
        large_draft_speed = results[1]["tokens_per_second"]  # 大草稿模型
        small_draft_speed = results[2]["tokens_per_second"]  # 小草稿模型
        svip_speed = results[3]["tokens_per_second"]  # SVIP策略
        
        print(f"\n{Fore.YELLOW}SVIP Strategy Analysis:{Style.RESET_ALL}")
        print(f"Speedup vs Autoregressive:   {svip_speed/baseline_speed:.2f}x")
        print(f"Speedup vs Large Draft Only: {svip_speed/large_draft_speed:.2f}x")
        print(f"Speedup vs Small Draft Only: {svip_speed/small_draft_speed:.2f}x")

if __name__ == "__main__":
    main()