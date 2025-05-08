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
    parser.add_argument("--approx_model_name", type=str, required=True, help="First approximate (small) model name")
    parser.add_argument("--approx_model_name2", type=str, help="Second approximate (small) model name")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--gamma", type=int, default=4, help="Number of tokens to speculate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

def load_model_and_tokenizer(model_name, device):
    """Load model and tokenizer from HuggingFace."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if "cuda" in device else torch.float32)
    model = model.to(device)
    model.eval()
    return model, tokenizer

def run_generation(method_name, input_ids, generate_func, *args, **kwargs):
    """Run generation with timing and statistics."""
    print(f"\n{Fore.CYAN}Running {method_name}...{Style.RESET_ALL}")
    
    with contexttimer.Timer() as timer:
        output_ids = generate_func(*args, **kwargs)
    
    # Calculate statistics
    num_new_tokens = output_ids.shape[1] - input_ids.shape[1]
    elapsed_time = timer.elapsed
    tokens_per_second = num_new_tokens / elapsed_time
    
    print(f"{Fore.GREEN}Generated {num_new_tokens} tokens in {elapsed_time:.2f} seconds")
    print(f"Performance: {tokens_per_second:.2f} tokens/s{Style.RESET_ALL}")
    
    # Return the results for comparison
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

def generate_speculative_single(input_ids, target_model, approx_model, max_tokens, gamma, temp, top_k, top_p, seed=None):
    """Speculative decoding with a single draft model."""
    return speculative_sampling(input_ids, approx_model, target_model, max_tokens, gamma, temp, top_k, top_p, random_seed=seed)

def generate_speculative_dual(input_ids, target_model, approx_model1, approx_model2, max_tokens, gamma, temp, top_k, top_p, seed=None):
    """Speculative decoding with two draft models."""
    return speculative_sampling(input_ids, approx_model1, target_model, max_tokens, gamma, temp, top_k, top_p, approx_model2=approx_model2, random_seed=seed)

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
        "Target + Approx Model 1 Speculative",
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
        args.seed
    ))
    
    # Method 3: Speculative decoding with target + approx_model2 (if available)
    if approx_model2:
        results.append(run_generation(
            "Target + Approx Model 2 Speculative",
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
            args.seed
        ))
    
    # Method 4: Speculative decoding with target + randomly selecting between approx_model1 and approx_model2
    if approx_model2:
        results.append(run_generation(
            "Target + Both Approx Models Random Speculative",
            input_ids,
            generate_speculative_dual,
            input_ids, 
            target_model, 
            approx_model1, 
            approx_model2, 
            args.max_tokens, 
            args.gamma, 
            args.temperature, 
            args.top_k, 
            args.top_p,
            args.seed
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
    print(f"{'Method':<40} {'Tokens':<10} {'Time (s)':<10} {'Tokens/s':<10}")
    print("-" * 80)
    for result in results:
        print(f"{result['method']:<40} {result['tokens']:<10.0f} {result['time']:<10.2f} {result['tokens_per_second']:<10.2f}")

if __name__ == "__main__":
    main()