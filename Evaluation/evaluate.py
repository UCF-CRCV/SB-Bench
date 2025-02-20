#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from io import BytesIO
from PIL import Image
import torch

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from transformers import AutoTokenizer
from huggingface_hub import login
from datasets import load_dataset

# Log in to Hugging Face Hub (insert your token)
login(token="")


def run_internvl2():
    """OpenGVLab/InternVL2-8B"""
    model_name = "OpenGVLab/InternVL2-8B"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        disable_mm_preprocessor_cache=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in stop_tokens]
    return llm, stop_token_ids

def run_llava_onevision_qwen2():
    """llava-hf/llava-onevision-qwen2-7b-ov-hf"""
    model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    llm = LLM(
        model=model_name,
        max_model_len=16384,
        disable_mm_preprocessor_cache=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    return llm, None

def run_molmo():
    """allenai/Molmo-7B-D-0924"""
    model_name = "allenai/Molmo-7B-D-0924"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        disable_mm_preprocessor_cache=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    return llm, None

def run_phi3_5():
    """microsoft/Phi-3.5-vision-instruct"""
    model_name = "microsoft/Phi-3.5-vision-instruct"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        disable_mm_preprocessor_cache=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    return llm, None

def run_qwen2_vl():
    """Qwen/Qwen2-VL-7B-Instruct"""
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        disable_mm_preprocessor_cache=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    return llm, None

def run_llama_3_2_11b_vision():
    """meta-llama/Llama-3.2-11B-Vision-Instruct"""
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        disable_mm_preprocessor_cache=True,
        enforce_eager=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    return llm, None


MODEL_MAP = {
    "InternVL2-8B": run_internvl2,
    "llava-onevision-qwen2-7b-ov-hf": run_llava_onevision_qwen2,
    "Molmo-7B-D-0924": run_molmo,
    "Phi-3.5-vision-instruct": run_phi3_5,
    "Qwen2-VL-7B-Instruct": run_qwen2_vl,
    "Llama-3.2-11B-Vision-Instruct": run_llama_3_2_11b_vision,
}


def build_prompt(model_name: str, context: str, question: str,
                 ans0: str, ans1: str, ans2: str, text_only: bool = False) -> str:
    """
    Constructs the text portion of the prompt for a single CSV row.
    """
    answers = 'Options: 1. ' + ans0 + ' 2. ' + ans1 + ' 3. ' + ans2
    prompt = context + '\n' + question + '\n' + answers
    post_prompt = "Give the output in strict JSON format. {\"answer\": int (1, 2, or 3) , \"reason\": \"Explain your reasoning here\"}"
    prompt_text = prompt + '\n' + post_prompt

    if "Qwen2-VL-7B-Instruct" in model_name:
        if text_only:
            prompt_text = "<|im_start|>user\n" + f"{prompt_text}<|im_end|>\n" + "<|im_start|>assistant\n"
        else:
            prompt_text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + f"{prompt_text}<|im_end|>\n" + "<|im_start|>assistant\n"
    elif "InternVL2-8B" in model_name:
        tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL2-8B', trust_remote_code=True)
        if text_only:
            messages = [{'role': 'user', 'content': f"\n{prompt_text}"}]
        else:
            messages = [{'role': 'user', 'content': f"<image>\n{prompt_text}"}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "llava-onevision-qwen2" in model_name:
        if text_only:
            prompt_text = f"<|im_start|>user \n{prompt_text}<|im_end|>" + "<|im_start|>assistant\n"
        else:
            prompt_text = f"<|im_start|>user <image>\n{prompt_text}<|im_end|>" + "<|im_start|>assistant\n"
    elif "Molmo-7B-D-0924" in model_name:
        prompt_text = prompt_text
    elif "Phi-3.5-vision-instruct" in model_name:
        if text_only:
            prompt_text = f"<|user|>\n{prompt_text}<|end|>\n<|assistant|>\n"
        else:
            prompt_text = f"<|user|>\n<|image_1|>\n{prompt_text}<|end|>\n<|assistant|>\n"
    elif "Llama-3.2-11B" in model_name:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-11B-Vision-Instruct')
        if text_only:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": f"{prompt_text}"}]
            }]
        else:
            messages = [{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": f"{prompt_text}"}]
            }]
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt_text


def save_results(results_data, output_path):
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(output_path, index=False)
    print(f"Inference done. Results saved to {output_path}")


def main(args):
    if args.bias_type != 'All':
        bias_types = [args.bias_type]
    else:
        bias_types = ['Age', 'Disability_status', 'Gender_identity', 'Physical_appearance',
                      'Sexual_orientation', 'Religion', 'Nationality', 'Race_ethnicity', 'SES']

    # Validate model name.
    if args.model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model '{args.model_name}'. Must be one of: {', '.join(MODEL_MAP.keys())}")

    # Create output directory.
    base_output_dir = os.path.join(args.output_path, f"tt_results_{args.model_name}")
    os.makedirs(base_output_dir, exist_ok=True)

    # Instantiate the chosen model.
    llm, stop_token_ids = MODEL_MAP[args.model_name]()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256, stop_token_ids=stop_token_ids)
    
    ds = load_dataset("ucf-crcv/SB-Bench", split="test")

    # Loop over each bias type.
    for bias in bias_types:

        print(f"> Evaluating {bias}...")

        filtered_ds = ds.filter(lambda example: example['category'] == bias)
        df = filtered_ds.to_pandas()

        batch_inputs = []
        record_map = []  
        results_data = []
        batch_processed = 0

        # Process each row.
        for row_idx, row in df.iterrows():
            uid = row.get('id', '')
            image = row.get('file_name', '')
            context = row.get('context', '')
            question = row.get('question', '')
            ans0 = row.get('ans0', '')
            ans1 = row.get('ans1', '')
            ans2 = row.get('ans2', '')
            label = row.get('label', '')

            prompt_str = build_prompt(args.model_name, context, question, ans0, ans1, ans2)

            image = Image.open(BytesIO(image['bytes']))
            image = image.resize((1920, 1024))

            batch_inputs.append({
                "prompt": prompt_str,
                "multi_modal_data": {"image": image},
            })

            record_map.append((uid, context, question, ans0, ans1, ans2, label))

            # Run inference on a batch.
            if ((row_idx + 1) % args.batch_size == 0) or (row_idx == df.shape[0] - 1):
                print(f"Running inference on {len(batch_inputs)} items...")
                outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
                batch_processed += 1
                batch_inputs = []

                for out_idx, output_obj in enumerate(outputs):
                    text_output = output_obj.outputs[0].text
                    rec_index = ((batch_processed - 1) * args.batch_size) + out_idx
                    if rec_index >= len(record_map):
                        continue

                    uid, context, question, ans0, ans1, ans2, label = record_map[rec_index]

                    results_data.append({
                        "id": uid,
                        "context": context,
                        "question": question,
                        "ans0": ans0,
                        "ans1": ans1,
                        "ans2": ans2,
                        "label": label,
                        "model_output": text_output
                    })

            output_path = os.path.join(args.output_path, f"tt_results_{args.model_name}", f"{bias}.csv")
            
        save_results(results_data, output_path)


def check_pytorch_gpu():
    try:
        if torch.cuda.is_available():
            print(f"PyTorch can access {torch.cuda.device_count()} GPU(s).")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("PyTorch cannot access any GPUs.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference for specific V+L models.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Which model to run. Must be one of: " + ", ".join(MODEL_MAP.keys()))
    parser.add_argument("--bias_type", type=str, default="All",
                        choices=['Age', 'Disability_status', 'Gender_identity', 'Physical_appearance',
                                 'Sexual_orientation', 'Religion', 'Nationality', 'Race_ethnicity', 'SES', 'All'],
                        help="Type of Bias to evaluate")
    parser.add_argument("--output_path", type=str, default="./",
                        help="File path to store inference outputs CSV.")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    check_pytorch_gpu()
    main(args)
