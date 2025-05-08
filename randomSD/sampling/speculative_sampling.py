import torch
from tqdm import tqdm
import torch

from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn
from globals import Decoder


def _truncate_logits(logits):
    """Truncate logits to expected vocab size for Qwen models"""
    QWEN_VOCAB_SIZE = 151643
    if logits.shape[-1] > QWEN_VOCAB_SIZE:
        return logits[..., :QWEN_VOCAB_SIZE]
    return logits


@torch.no_grad()
def speculative_sampling(prefix: torch.Tensor, approx_model: torch.nn.Module, target_model: torch.nn.Module,
                         max_len: int, gamma: int = 4,
                         temperature: float = 1, top_k: int = 0, top_p: float = 0, verbose: bool = False,
                         random_seed: int = None,
                         approx_model2: torch.nn.Module = None) -> torch.Tensor:
    """
    支持两个draft model，每个draft token随机选择一个draft model生成。
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len

    assert prefix.shape[0] == 1, "input batch size must be 1"
    assert approx_model.device == target_model.device
    device = target_model.device

    # 支持两个draft model
    approx_model_cache1 = KVCacheModel(approx_model, temperature, top_k, top_p)
    approx_model_cache2 = KVCacheModel(approx_model2, temperature, top_k, top_p) if approx_model2 is not None else None
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)

    resample_count = 0
    target_sample_count = 0
    accepted_count = 0

    while prefix.shape[1] < T:
        prefix_len = prefix.shape[1]
        x = prefix
        draft_token_sources = []  # 记录每个draft token是哪个draft model生成的
        # draft阶段，每个token随机选择draft model生成
        for i in range(gamma):
            if approx_model2 is not None:
                if random_seed is not None:
                    torch.manual_seed(random_seed + i)
                model_choice = torch.randint(0, 2, (1,)).item()
                if model_choice == 0:
                    q = approx_model_cache1._forward_with_kvcache(x, use_debug=False)
                    next_tok = sample(q)
                    draft_token_sources.append(1)
                else:
                    q = approx_model_cache2._forward_with_kvcache(x, use_debug=False)
                    next_tok = sample(q)
                    draft_token_sources.append(2)
            else:
                q = approx_model_cache1._forward_with_kvcache(x, use_debug=False)
                next_tok = sample(q)
                draft_token_sources.append(1)
            x = torch.cat((x, next_tok), dim=1)
        # target model forward
        _ = target_model_cache.generate(x, 1)
        n = prefix_len + gamma - 1

        # Truncate logits for probability comparison
        target_probs = _truncate_logits(target_model_cache._prob_history)
        approx_probs1 = _truncate_logits(approx_model_cache1._prob_history)
        approx_probs2 = _truncate_logits(approx_model_cache2._prob_history) if approx_model2 is not None else None

        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed + i)
            r = torch.rand(1, device=device)
            j = x[:, prefix_len + i]
            # 选择当前token用哪个draft model的概率
            if approx_model2 is not None:
                if draft_token_sources[i] == 1:
                    approx_probs = approx_probs1
                else:
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

        approx_model_cache1.rollback(n + 1)
        if approx_model2 is not None:
            approx_model_cache2.rollback(n + 1)
        assert approx_model_cache1._prob_history.shape[
                   -2] <= n + 1, f"approx_model prob list shape {approx_model_cache1._prob_history.shape}, n {n}"
        if approx_model2 is not None:
            assert approx_model_cache2._prob_history.shape[
                       -2] <= n + 1, f"approx_model2 prob list shape {approx_model_cache2._prob_history.shape}, n {n}"

        if n < prefix_len + gamma - 1:
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_probs[:, n, :]))
            if verbose:
                print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n + 1)
        else:
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n + 2)
        prefix = torch.cat((prefix, t), dim=1)
        # 检查是否生成了EOS token
        if t[0] == target_model.config.eos_token_id:
            break

    if verbose:
        print(
            f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    return prefix


@torch.no_grad()
def speculative_sampling_v2(prefix: torch.Tensor, approx_model: torch.nn.Module, target_model: torch.nn.Module,
                            max_len: int, gamma: int = 4,
                            temperature: float = 1, top_k: int = 0, top_p: float = 0,
                            random_seed: int = None) -> torch.Tensor:
    seq_len = prefix.shape[1]
    T = seq_len + max_len

    assert prefix.shape[0] == 1, "input batch size must be 1"

    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :],
                                              temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)

            # normalize the logits
            for i in range(q.shape[1]):
                q[:, i, :] = _truncate_logits(norm_logits(q[:, i, :],
                                                          temperature, top_k, top_p))
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = target_model(x).logits
            for i in range(p.shape[1]):
                p[:, i, :] = _truncate_logits(norm_logits(p[:, i, :],
                                                          temperature, top_k, top_p))

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)

            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device=p.device)
                j = x[:, prefix_len + i]

                if r < torch.min(torch.tensor([1], device=q.device),
                                 p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                else:
                    # reject
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    is_all_accept = False
                    break

            prefix = x[:, :n + 1]

            if is_all_accept:
                t = sample(p[:, -1, :])

            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(n - pbar.n)

    return prefix

