import torch
import time
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
from typing import List, Dict
import numpy as np

class LlamaKVCacheTimer:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.k_times = []
        self.v_times = []
        self.layer_indices = []
        self.current_k_start = None
        self.current_v_start = None
        self.active = False
        
        for i, layer in enumerate(model.model.layers):
            self.hooks.append(layer.self_attn.k_proj.register_forward_pre_hook(
                lambda module, input, layer_idx=i: self._k_pre_hook(module, input, layer_idx)
            ))
            self.hooks.append(layer.self_attn.k_proj.register_forward_hook(
                lambda module, input, output, layer_idx=i: self._k_post_hook(module, input, output, layer_idx)
            ))
            
            self.hooks.append(layer.self_attn.v_proj.register_forward_pre_hook(
                lambda module, input, layer_idx=i: self._v_pre_hook(module, input, layer_idx)
            ))
            self.hooks.append(layer.self_attn.v_proj.register_forward_hook(
                lambda module, input, output, layer_idx=i: self._v_post_hook(module, input, output, layer_idx)
            ))
    
    def _k_pre_hook(self, module, input, layer_idx):
        if self.active and len(input[0].shape) == 3:  
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.current_k_start = time.time()
    
    def _k_post_hook(self, module, input, output, layer_idx):
        if self.active and self.current_k_start is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - self.current_k_start
            self.k_times.append(elapsed)
            self.layer_indices.append(layer_idx)
            self.current_k_start = None
    
    def _v_pre_hook(self, module, input, layer_idx):
        if self.active and len(input[0].shape) == 3: 
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.current_v_start = time.time()
    
    def _v_post_hook(self, module, input, output, layer_idx):
        if self.active and self.current_v_start is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - self.current_v_start
            self.v_times.append(elapsed)
            self.current_v_start = None
    
    def start_timing(self):
        self.k_times = []
        self.v_times = []
        self.layer_indices = []
        self.active = True
    
    def stop_timing(self):
        self.active = False
    
    def get_total_kv_time(self):
        return (sum(self.k_times) + sum(self.v_times)) * 1000
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def measure_llama_kv_cache_times(model_name="lmsys/vicuna-7b-v1.3", num_tokens=1000, prompt="Once upon a time"):
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    print("Setting up KV cache timer...")
    timer = LlamaKVCacheTimer(model)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    print("Initializing KV cache...")
    with torch.no_grad():
        outputs = model(input_ids)
        past_key_values = outputs.past_key_values
    
    kv_cache_times = []
    
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
    
    print("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = model(next_token, past_key_values=past_key_values, use_cache=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    print(f"Start measuring KV cache computation time for {num_tokens} tokens...")
    for i in range(num_tokens):
        timer.start_timing()
        
        with torch.no_grad():
            outputs = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        timer.stop_timing()
        
        kv_time = timer.get_total_kv_time()
        kv_cache_times.append(kv_time)
        
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
        
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{num_tokens} tokens")
    
    timer.remove_hooks()
    
    return kv_cache_times

def plot_detailed_kv_cache_times(kv_cache_times):
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(kv_cache_times) + 1), kv_cache_times, 'b-', alpha=0.7)
    
    x = np.array(range(1, len(kv_cache_times) + 1))
    z = np.polyfit(x, kv_cache_times, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r--', linewidth=2, label=f'Trend line: {z[0]:.6f}x + {z[1]:.6f}')
    
    plt.title('KV Cache Computation Time per Token vicuna')
    plt.xlabel('Token Position')
    plt.ylabel('Time (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    window_size = min(50, len(kv_cache_times) // 10) if len(kv_cache_times) > 50 else 5
    moving_avg = np.convolve(kv_cache_times, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size, window_size + len(moving_avg)), moving_avg, 'g-')
    plt.title(f'{window_size}-point Moving Average vicuna')
    plt.xlabel('Token Position')
    plt.ylabel('Time (ms)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kv_cache_detailed_time_plot.png', dpi=300)
    
    plt.figure(figsize=(10, 6))
    plt.hist(kv_cache_times, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('KV Cache Computation Time Distribution')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('kv_cache_time_histogram.png', dpi=300)
    
    plt.show()
    
    print("\nKV Cache Computation Time Statistics:")
    print(f"Mean time: {np.mean(kv_cache_times):.4f} ms")
    print(f"Median time: {np.median(kv_cache_times):.4f} ms")
    print(f"Std: {np.std(kv_cache_times):.4f} ms")
    print(f"Min time: {np.min(kv_cache_times):.4f} ms")
    print(f"Max time: {np.max(kv_cache_times):.4f} ms")
    print(f"Total time: {np.sum(kv_cache_times):.4f} ms")
    
    with open('kv_cache_times.txt', 'w') as f:
        f.write("Token Position,KV Cache Computation Time(ms)\n")
        for i, time_val in enumerate(kv_cache_times):
            f.write(f"{i+1},{time_val:.6f}\n")
    
    np.save('kv_cache_times.npy', np.array(kv_cache_times))

if __name__ == "__main__":
    print("Start measuring LLaMA model KV cache computation time...")
    
    model_name = "lmsys/vicuna-7b-v1.3"
    num_tokens = 1000
    prompt = "Once upon a time, "
    
    kv_cache_times = measure_llama_kv_cache_times(model_name, num_tokens, prompt)
    
    plot_detailed_kv_cache_times(kv_cache_times)
    
    print("Measurement complete. Results have been saved to files and plots.")