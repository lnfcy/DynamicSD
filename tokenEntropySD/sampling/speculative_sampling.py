import torch
from tqdm import tqdm

from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn
from globals import Decoder


def _truncate_logits(logits):
    """Truncate logits to expected vocab size for Qwen models"""
    if logits is None:
        return None
        
    QWEN_VOCAB_SIZE = 151643
    if logits.shape[-1] > QWEN_VOCAB_SIZE:
        return logits[..., :QWEN_VOCAB_SIZE]
    return logits


@torch.no_grad()
def calc_entropy(logits):
    """计算logits的熵值"""
    # logits: (batch, vocab)
    assert len(logits.shape) == 2
    # 数值稳定化
    logits = logits - logits.max(dim=1).values.unsqueeze(1)
    # 计算softmax概率
    exp_logits = torch.exp(logits)
    probs = exp_logits / exp_logits.sum(dim=1).unsqueeze(1)
    # 计算熵: -sum(p * log(p))
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1)


@torch.no_grad()
def speculative_sampling(prefix: torch.Tensor, approx_model: torch.nn.Module, target_model: torch.nn.Module,
                         max_len: int, gamma: int = 4,
                         temperature: float = 1, top_k: int = 0, top_p: float = 0, verbose: bool = False,
                         random_seed: int = None,
                         approx_model2: torch.nn.Module = None, entropy_threshold: float = 0.3,
                         use_svip: bool = False) -> torch.Tensor:
    """
    
    策略选择：
    - 当approx_model2为None时，只使用单个draft model
    - 当approx_model2不为None时：
    - 如果use_svip=True，使用基于熵的自适应策略（熵高用大模型approx_model，熵低用小模型approx_model2）
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len

    assert prefix.shape[0] == 1, "input batch size must be 1"
    assert approx_model.device == target_model.device
    
    if use_svip:
        assert approx_model2 is not None, "SVIP mode requires two draft models"
        
    device = target_model.device

    approx_model_cache1 = KVCacheModel(approx_model, temperature, top_k, top_p)
    approx_model_cache2 = KVCacheModel(approx_model2, temperature, top_k, top_p) if approx_model2 is not None else None
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)

    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    
    # 记录当前熵值，初始设为0
    current_entropy = torch.zeros(1, device=device)

    # 跟踪approx_model_cache2是否已被使用
    approx_model2_used = False

    while prefix.shape[1] < T:
        prefix_len = prefix.shape[1]
        x = prefix
        draft_token_sources = []  # 记录每个draft token是哪个draft model生成的
        # draft阶段，每个token根据策略选择draft model生成
        for i in range(gamma):
            if approx_model2 is not None:
                if use_svip:
                    # 使用SVIP熵自适应策略
                    if i == 0:
                        # 第一个token默认使用大模型
                        model_choice = 0
                    else:
                        # 应用SVIP源代码的熵变换公式
                        transformed_entropy = torch.sqrt(current_entropy * 0.15)
                        # 熵高于阈值用大模型，低于阈值用小模型
                        model_choice = 0 if transformed_entropy > entropy_threshold else 1
                    
                    if verbose and i > 0:
                        print(f"Position {i}: Using {'large' if model_choice == 0 else 'small'} draft model (entropy: {transformed_entropy.item():.4f})")
                
                if model_choice == 0:
                    q = approx_model_cache1._forward_with_kvcache(x, use_debug=False)
                    next_tok = sample(q)
                    draft_token_sources.append(1)
                else:
                    q = approx_model_cache2._forward_with_kvcache(x, use_debug=False)
                    next_tok = sample(q)
                    draft_token_sources.append(2)
                    approx_model2_used = True
            else:
                q = approx_model_cache1._forward_with_kvcache(x, use_debug=False)
                next_tok = sample(q)
                draft_token_sources.append(1)
            
            # 更新熵值，用于下一个token的决策
            current_entropy = calc_entropy(q)
            
            x = torch.cat((x, next_tok), dim=1)
        _ = target_model_cache.generate(x, 1)
        n = prefix_len + gamma - 1

        target_probs = _truncate_logits(target_model_cache._prob_history)
        approx_probs1 = _truncate_logits(approx_model_cache1._prob_history)
        # 增加安全检查，确保approx_model_cache2存在且有_prob_history
        approx_probs2 = None
        if approx_model2 is not None and approx_model_cache2 is not None and hasattr(approx_model_cache2, '_prob_history'):
            approx_probs2 = _truncate_logits(approx_model_cache2._prob_history)

        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed + i)
            r = torch.rand(1, device=device)
            j = x[:, prefix_len + i]
            # 选择当前token用哪个draft model的概率
            if approx_model2 is not None and draft_token_sources[i] == 2 and approx_probs2 is not None:
                approx_probs = approx_probs2
            else:
                approx_probs = approx_probs1
            # 验证
            if r > (target_probs[:, prefix_len + i - 1, j]) / (approx_probs[:, prefix_len + i - 1, j]):
                n = prefix_len + i - 1
                break
            if verbose:
                print(
                    f"approx guess accepted {j[0]} (draft{draft_token_sources[i]}): \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")
            accepted_count += 1

        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        # rollback所有draft cache到n+1
        approx_model_cache1.rollback(n + 1)
        
        # 只有在approx_model_cache2被使用过且有_past_key_values属性时才回滚
        if (approx_model2 is not None and approx_model_cache2 is not None and 
            approx_model2_used and hasattr(approx_model_cache2, '_past_key_values') and 
            approx_model_cache2._past_key_values):
            approx_model_cache2.rollback(n + 1)
            
        assert approx_model_cache1._prob_history.shape[
                   -2] <= n + 1, f"approx_model prob list shape {approx_model_cache1._prob_history.shape}, n {n}"
                   
        if (approx_model2 is not None and approx_model_cache2 is not None and 
            hasattr(approx_model_cache2, '_prob_history') and approx_model_cache2._prob_history is not None):
            assert approx_model_cache2._prob_history.shape[
                       -2] <= n + 1, f"approx_model2 prob list shape {approx_model_cache2._prob_history.shape}, n {n}"

        if n < prefix_len + gamma - 1:
            # 确定使用哪个模型的概率进行重采样
            approx_probs = approx_probs1
            if (approx_model2 is not None and n - prefix_len + 1 < len(draft_token_sources) and
                draft_token_sources[n - prefix_len + 1] == 2 and approx_probs2 is not None):
                approx_probs = approx_probs2
                
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_probs[:, n, :]))
            if verbose:
                print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n + 1)
            
            # 更新当前熵值用于下一轮
            current_entropy = calc_entropy(target_model_cache._prob_history[:, n, :])
        else:
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n + 2)
            
            # 更新当前熵值用于下一轮
            current_entropy = calc_entropy(target_model_cache._prob_history[:, -1, :])
            
        prefix = torch.cat((prefix, t), dim=1)
        # 检查是否生成了EOS token
        if t[0] == target_model.config.eos_token_id:
            break

    if verbose:
        print(
            f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    return prefix

