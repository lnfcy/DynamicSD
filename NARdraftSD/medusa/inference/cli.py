import argparse
import os
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from fastchat.serve.cli import SimpleChatIO, RichChatIO, ProgrammaticChatIO
from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template
import json
from medusa.model.medusa_model import MedusaModel
import time
import random
from colorama import Fore, Style


class CustomChatIO(SimpleChatIO):
    def stream_output(self, output_text):
        print(output_text, end="", flush=True)


@torch.no_grad()
def standard_speculative_sampling(
        input_ids,
        target_model,
        draft_model,
        tokenizer,
        max_steps=512,
        gamma=4,
        temperature=0.7,
        device="cuda",
):
    output_ids = input_ids.clone().to(device)
    start_pos = output_ids.shape[1]
    last_print_len = 0
    token_stats = []

    for step in range(max_steps):
        prefix_len = output_ids.shape[1]
        x = output_ids
        
        for _ in range(gamma):
            with torch.inference_mode():
                draft_output = draft_model(
                    input_ids=x,
                    use_cache=True,
                )
                draft_logits = draft_output.logits[:, -1, :]
                next_token = torch.argmax(draft_logits, dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)
            
        with torch.no_grad():
            target_outputs = target_model(
                input_ids=x,
                use_cache=True,
            )
            target_logits = target_outputs.logits
            
        n = prefix_len - 1
        is_all_accept = True
        accepted_colors = []
        
        for i in range(gamma):
            j = x[:, prefix_len + i]
            with torch.inference_mode():
                draft_outputs = draft_model(
                    input_ids=x[:, :prefix_len + i + 1],
                    use_cache=True,
                )
                draft_logits = draft_outputs.logits[:, -1, :]
                draft_probs = F.softmax(draft_logits, dim=-1)
                target_probs = F.softmax(target_logits[:, prefix_len + i - 1, :], dim=-1)
                
            r = torch.rand(1, device=target_logits.device)
            if r > (target_probs[0, j] / draft_probs[0, j]):
                n = prefix_len + i - 1
                is_all_accept = False
                break
            else:
                n = prefix_len + i
                accepted_colors.append("blue")
                
        output_ids = x[:, :n + 1]
        
        if not is_all_accept:
            resample_logits = target_logits[:, n, :] - draft_probs
            resample_token = torch.argmax(resample_logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, resample_token], dim=1)
            accepted_colors.append("green")
        else:
            next_logits = target_logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, next_token], dim=1)
            accepted_colors.append("green")
            
        tokens = output_ids[0, start_pos:]
        for color, token_id in zip(accepted_colors, tokens[-len(accepted_colors):]):
            token_text = tokenizer.decode([token_id], skip_special_tokens=True, spaces_between_special_tokens=False)
            token_stats.append((token_text, color))
            
        decoded = tokenizer.decode(
            output_ids[0, start_pos:], skip_special_tokens=True, spaces_between_special_tokens=False
        )
        if last_print_len < len(decoded):
            new_text = decoded[last_print_len:]
            yield new_text
            last_print_len = len(decoded)
            
        if (
                hasattr(tokenizer, "eos_token_id")
                and output_ids[0, -1].item() == tokenizer.eos_token_id
        ):
            break
            
    yield ("__COLOR_STATS__", token_stats)


@torch.no_grad()
def dynamic_speculative_sampling_with_medusa_lookahead(
        input_ids,
        target_model,
        draft_model1,
        draft_model2,
        tokenizer,
        max_steps=512,
        low_confidence_threshold=0.8,  # 用于区分简单和中等token
        high_confidence_threshold=0.95,  # 用于确定是否提前插入预测token
        gamma=4,
        device="cuda",
):
    output_ids = input_ids.clone().to(device)
    draft_models = [draft_model1, draft_model2]
    start_pos = output_ids.shape[1]
    last_print_len = 0
    token_stats = []
    medusa_head_idx = 1
    
    for step in range(max_steps):
        prefix_len = output_ids.shape[1]
        x = output_ids
        draft_token_sources = []
        lookahead_confidences = []
        prefilled_positions = set()  # 记录已经被提前填充的位置
        
        # 第一步：使用Medusa head进行前瞻并可能提前插入token
        medusa_logits = draft_models[1](  # 使用draft_model2的medusa head
            input_ids=x,
            medusa_forward=True,
        )
        
        if medusa_logits.shape[0] > medusa_head_idx:
            for pos in range(gamma):
                if pos < medusa_logits.shape[0]:
                    next_token_logits = medusa_logits[pos, :, -1, :]
                    next_token_probs = F.softmax(next_token_logits, dim=-1)
                    confidence, predicted_token = next_token_probs.max(dim=-1)
                    
                    # 修改逻辑1: 低置信度时使用预填充
                    if confidence.item() < low_confidence_threshold:
                        prefilled_positions.add(prefix_len + pos)
                        if pos == 0:
                            x = torch.cat([x, predicted_token.unsqueeze(0).unsqueeze(0)], dim=1)
                            draft_token_sources.append("prefilled")
                            lookahead_confidences.append(confidence.item())
        
        # 第二步：对未填充的位置进行正常的动态推测解码
        current_len = x.shape[1]
        while current_len < prefix_len + gamma:
            pos = current_len - prefix_len
            if current_len not in prefilled_positions:
                # 使用当前token的Medusa head预测下一个token的置信度
                medusa_logits = draft_models[1](
                    input_ids=x,
                    medusa_forward=True,
                )
                
                if medusa_logits.shape[0] > medusa_head_idx:
                    next_next_token_logits = medusa_logits[medusa_head_idx, :, -1, :]
                    next_next_token_probs = F.softmax(next_next_token_logits, dim=-1)
                    lookahead_confidence, _ = next_next_token_probs.max(dim=-1)
                    lookahead_confidences.append(lookahead_confidence.item())
                    
                    # 修改逻辑2: 修改模型选择逻辑
                    if lookahead_confidence.item() > high_confidence_threshold:
                        current_draft_idx = 1  # 高置信度使用小模型
                    else:
                        current_draft_idx = 0  # 中等置信度使用大模型
                else:
                    lookahead_confidences.append(None)
                    current_draft_idx = 0
                
                current_draft_model = draft_models[current_draft_idx]
                draft_outputs = current_draft_model(
                    input_ids=x,
                    use_cache=True,
                )
                logits = draft_outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                x = torch.cat([x, next_token], dim=1)
                draft_token_sources.append(current_draft_idx)
            
            current_len += 1
        
        # 第三步：使用target model进行验证
        target_outputs = target_model(
            input_ids=x,
            use_cache=True,
        )
        target_logits = target_outputs.logits
        
        n = prefix_len - 1
        is_all_accept = True
        accepted_colors = []
        
        for i in range(gamma):
            if prefix_len + i in prefilled_positions:
                # 对预填充的token进行验证
                j = x[:, prefix_len + i]
                target_probs = F.softmax(target_logits[:, prefix_len + i - 1, :], dim=-1)
                draft_probs = torch.zeros_like(target_probs)
                draft_probs[0, j] = 1.0  # 使用one-hot分布
                
                r = torch.rand(1, device=target_logits.device)
                if r > target_probs[0, j]:
                    n = prefix_len + i - 1
                    is_all_accept = False
                    break
                else:
                    n = prefix_len + i
                    accepted_colors.append("purple")  # 使用紫色表示预填充token
            else:
                # 对draft model生成的token进行验证
                j = x[:, prefix_len + i]
                draft_outputs = draft_models[draft_token_sources[i]](
                    input_ids=x[:, :prefix_len + i + 1],
                    use_cache=True,
                )
                draft_logits = draft_outputs.logits[:, -1, :]
                draft_probs = F.softmax(draft_logits, dim=-1)
                target_probs = F.softmax(target_logits[:, prefix_len + i - 1, :], dim=-1)
                
                r = torch.rand(1, device=target_logits.device)
                if r > (target_probs[0, j] / draft_probs[0, j]):
                    n = prefix_len + i - 1
                    is_all_accept = False
                    break
                else:
                    n = prefix_len + i
                    if draft_token_sources[i] == 0:
                        accepted_colors.append("red")
                    else:
                        accepted_colors.append("blue")
        
        output_ids = x[:, :n + 1]
        
        if not is_all_accept:
            resample_logits = target_logits[:, n, :] - draft_probs
            resample_token = torch.argmax(resample_logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, resample_token], dim=1)
            accepted_colors.append("green")
        else:
            next_logits = target_logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, next_token], dim=1)
            accepted_colors.append("green")

        tokens = output_ids[0, start_pos:]
        for color, token_id in zip(accepted_colors, tokens[-len(accepted_colors):]):
            token_text = tokenizer.decode([token_id], skip_special_tokens=True, spaces_between_special_tokens=False)
            token_stats.append((token_text, color))

        decoded = tokenizer.decode(
            output_ids[0, start_pos:], skip_special_tokens=True, spaces_between_special_tokens=False
        )
        if last_print_len < len(decoded):
            new_text = decoded[last_print_len:]
            yield new_text
            last_print_len = len(decoded)

        if draft_token_sources and lookahead_confidences:
            debug_info = f"步骤 {step}，模型选择: {draft_token_sources}, 前瞻置信度: {[f'{c:.4f}' if c is not None else 'N/A' for c in lookahead_confidences]}"
            print(debug_info, file=sys.stderr)

        if (
                hasattr(tokenizer, "eos_token_id")
                and output_ids[0, -1].item() == tokenizer.eos_token_id
        ):
            break

    yield ("__COLOR_STATS__", token_stats)


@torch.no_grad()
def autoregressive_sampling(
        input_ids,
        target_model,
        tokenizer,
        max_steps=512,
        device="cuda",
):
    output_ids = input_ids.clone().to(device)
    start_pos = output_ids.shape[1]
    last_print_len = 0

    for step in range(max_steps):
        with torch.inference_mode():
            output = target_model(output_ids)
            next_token_logits = output.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        output_ids = torch.cat([output_ids, next_token], dim=1)
        
        decoded = tokenizer.decode(
            output_ids[0, start_pos:], skip_special_tokens=True, spaces_between_special_tokens=False
        )
        if last_print_len < len(decoded):
            new_text = decoded[last_print_len:]
            yield new_text
            last_print_len = len(decoded)
            
        if (
                hasattr(tokenizer, "eos_token_id")
                and output_ids[0, -1].item() == tokenizer.eos_token_id
        ):
            break


def main(args):
    if args.style == "simple":
        chatio = CustomChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        from transformers import AutoModelForCausalLM
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }
        if args.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if args.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        target_model = AutoModelForCausalLM.from_pretrained(
            args.target_model,
            **model_kwargs,
        )
        print(f"target_model class: {type(target_model)}")
        draft_model1 = MedusaModel.from_pretrained(
            args.draft_model1,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        draft_model2 = MedusaModel.from_pretrained(
            args.draft_model2,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(args.target_model)
        conv = None

        def new_chat():
            return get_conversation_template(args.target_model)

        def reload_conv(conv):
            for message in conv.messages[conv.offset:]:
                chatio.prompt_for_output(message[0])
                chatio.print_output(message[1])

        confidence_threshold = args.confidence_threshold  # 新增变量，初始为参数值

        while True:
            if not conv:
                conv = new_chat()

            # ===== 新增：每次对话前询问置信度 =====
            if args.mode == 2:
                try:
                    low_input = input(f"当前低置信度阈值为 {args.low_confidence_threshold:.3f}，如需修改请输入新值（回车跳过）：")
                    if low_input.strip():
                        args.low_confidence_threshold = float(low_input.strip())
                    
                    high_input = input(f"当前高置信度阈值为 {args.high_confidence_threshold:.3f}，如需修改请输入新值（回车跳过）：")
                    if high_input.strip():
                        args.high_confidence_threshold = float(high_input.strip())
                except Exception as e:
                    print(f"输入无效，继续使用当前阈值")

            try:
                inp = chatio.prompt_for_input(conv.roles[0])
            except EOFError:
                inp = ""

            if inp == "!!exit" or not inp:
                print("exit...")
                break
            elif inp == "!!reset":
                print("resetting...")
                conv = new_chat()
                continue
            elif inp == "!!remove":
                print("removing last message...")
                if len(conv.messages) > conv.offset:
                    if conv.messages[-1][0] == conv.roles[1]:
                        conv.messages.pop()
                    if conv.messages[-1][0] == conv.roles[0]:
                        conv.messages.pop()
                    reload_conv(conv)
                else:
                    print("No messages to remove.")
                continue
            elif inp == "!!regen":
                print("regenerating last message...")
                if len(conv.messages) > conv.offset:
                    if conv.messages[-1][0] == conv.roles[1]:
                        conv.messages.pop()
                    if conv.messages[-1][0] == conv.roles[0]:
                        reload_conv(conv)
                        inp = conv.messages.pop()[1]
                    else:
                        print("No user message to regenerate from.")
                        continue
                else:
                    print("No messages to regenerate.")
                    continue
            elif inp.startswith("!!save"):
                args = inp.split(" ", 1)

                if len(args) != 2:
                    print("usage: !!save <filename>")
                    continue
                else:
                    filename = args[1]

                if not "." in filename:
                    filename += ".json"

                print("saving...", filename)
                with open(filename, "w") as outfile:
                    json.dump(conv.dict(), outfile)
                continue
            elif inp.startswith("!!load"):
                args = inp.split(" ", 1)

                if len(args) != 2:
                    print("usage: !!load <filename>")
                    continue
                else:
                    filename = args[1]

                if not os.path.exists(filename):
                    if (not filename.endswith(".json")) and os.path.exists(
                            filename + ".json"
                    ):
                        filename += ".json"
                    else:
                        print("file not found:", filename)
                        continue

                print("loading...", filename)
                with open(filename, "r") as infile:
                    new_conv = json.load(infile)

                conv = get_conv_template(new_conv["template_name"])
                conv.set_system_message(new_conv["system_message"])
                conv.messages = new_conv["messages"]
                reload_conv(conv)
                continue

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            try:
                chatio.prompt_for_output(conv.roles[1])
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                    target_model.device
                )
                outputs = ""
                color_stats = None
                start_time = time.time()

                if args.mode == 0:
                    for new_token_text in standard_speculative_sampling(
                            input_ids=input_ids,
                            target_model=target_model,
                            draft_model=draft_model1,
                            tokenizer=tokenizer,
                            max_steps=args.max_steps,
                            gamma=4,
                    ):
                        if isinstance(new_token_text, tuple) and new_token_text[0] == "__COLOR_STATS__":
                            color_stats = new_token_text[1]
                        else:
                            print(new_token_text, end="", flush=True)
                            outputs += new_token_text
                    conv.update_last_message(outputs.strip())

                    if color_stats:
                        print("\n\nToken来源统计：")
                        for token, color in color_stats:
                            if color == "blue":
                                print(Fore.BLUE + token + Style.RESET_ALL, end="")
                            elif color == "green":
                                print(Fore.GREEN + token + Style.RESET_ALL, end="")
                            else:
                                print(token, end="")
                        print(Style.RESET_ALL)

                elif args.mode == 1:
                    for new_token_text in standard_speculative_sampling(
                            input_ids=input_ids,
                            target_model=target_model,
                            draft_model=draft_model2,
                            tokenizer=tokenizer,
                            max_steps=args.max_steps,
                            gamma=4,
                    ):
                        if isinstance(new_token_text, tuple) and new_token_text[0] == "__COLOR_STATS__":
                            color_stats = new_token_text[1]
                        else:
                            print(new_token_text, end="", flush=True)
                            outputs += new_token_text
                    conv.update_last_message(outputs.strip())

                    if color_stats:
                        print("\n\nToken来源统计：")
                        for token, color in color_stats:
                            if color == "blue":
                                print(Fore.BLUE + token + Style.RESET_ALL, end="")
                            elif color == "green":
                                print(Fore.GREEN + token + Style.RESET_ALL, end="")
                            else:
                                print(token, end="")
                        print(Style.RESET_ALL)

                elif args.mode == 2:
                    for new_token_text in dynamic_speculative_sampling_with_medusa_lookahead(
                            input_ids=input_ids,
                            target_model=target_model,
                            draft_model1=draft_model1,
                            draft_model2=draft_model2,
                            tokenizer=tokenizer,
                            max_steps=args.max_steps,
                            low_confidence_threshold=args.low_confidence_threshold,  # 使用当前置信度
                            high_confidence_threshold=args.high_confidence_threshold,  # 使用当前置信度
                            gamma=4,
                    ):
                        if isinstance(new_token_text, tuple) and new_token_text[0] == "__COLOR_STATS__":
                            color_stats = new_token_text[1]
                        else:
                            print(new_token_text, end="", flush=True)
                            outputs += new_token_text
                    conv.update_last_message(outputs.strip())

                    if color_stats:
                        print("\n\nToken来源统计：")
                        for token, color in color_stats:
                            if color == "red":
                                print(Fore.RED + token + Style.RESET_ALL, end="")
                            elif color == "blue":
                                print(Fore.BLUE + token + Style.RESET_ALL, end="")
                            elif color == "green":
                                print(Fore.GREEN + token + Style.RESET_ALL, end="")
                            elif color == "purple":
                                print(Fore.MAGENTA + token + Style.RESET_ALL, end="")
                            else:
                                print(token, end="")
                        print(Style.RESET_ALL)
                else:
                    for new_token_text in autoregressive_sampling(
                            input_ids=input_ids,
                            target_model=target_model,
                            tokenizer=tokenizer,
                            max_steps=args.max_steps,
                    ):
                        print(new_token_text, end="", flush=True)
                        outputs += new_token_text
                    conv.update_last_message(outputs.strip())

                end_time = time.time()
                print(f"\n生成整个回答的时间是: {end_time - start_time:.2f} 秒")

            except KeyboardInterrupt:
                print("stopped generation.")
                if conv.messages[-1][1] is None:
                    conv.messages.pop()
                    if conv.messages[-1][0] == conv.roles[0]:
                        conv.messages.pop()

                    reload_conv(conv)

    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name or path (deprecated, use --target-model instead)")
    parser.add_argument(
        "--target-model", type=str, help="Target model name or path"
    )
    parser.add_argument(
        "--draft-model1", type=str, help="First draft model name or path (larger model, e.g. 13B)"
    )
    parser.add_argument(
        "--draft-model2", type=str, help="Second draft model name or path (smaller model, e.g. 7B)"
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.8,
                        help="Confidence threshold for switching draft models")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.8,
                        help="Low confidence threshold for switching between draft models")
    parser.add_argument("--high-confidence-threshold", type=float, default = 0.95,
                        help="High confidence threshold for early token insertion")
    parser.add_argument("--mode", type=int, default=2, 
                       help="0: 标准SD (target_model + draft_model1), 1: 标准SD (target_model + draft_model2), 2: Medusa前瞻的动态SD, 4: 纯自回归")
    args = parser.parse_args()

    if args.model and not args.target_model:
        args.target_model = args.model

    if not args.target_model:
        parser.error("Either --model or --target-model must be specified")

    if bool(args.draft_model1) != bool(args.draft_model2):
        parser.error("Either both or neither of --draft-model1 and --draft-model2 must be specified")

    main(args)