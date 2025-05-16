#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from io import BytesIO
import torch
import base64
import re
import json
import base64
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAXBLOCK = 2**30
from io import BytesIO
from glob import glob

from vllm import LLM, SamplingParams
from huggingface_hub import login
from transformers import AutoProcessor
from transformers import AutoTokenizer
from qwen_vl_utils import process_vision_info

login(token="YOUR HF TOKEN HERE")

def get_qwen_omni(parameters='7B', version='2.5'):
    model_name = "Qwen/Qwen2.5-Omni-7B"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        disable_mm_preprocessor_cache=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_name)

    return llm, None, processor


def get_qwen_vl(parameters='7B', version='2.5'):
    """Qwen/Qwen2.5-VL-7B-Instruct"""
    if parameters=='default':
        parameters = '7B'
    if version=='default':
        version = '2.5'

    if version=='2.5':
        model_name = f"Qwen/Qwen2.5-VL-{parameters}-Instruct"
    elif version=='2':
        model_name = f"Qwen/Qwen2-VL-{parameters}-Instruct"
    else:
        raise NotImplementedError
    
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        disable_mm_preprocessor_cache=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_name)

    return llm, None, processor

def get_gemma3(parameters='12b', version='3'):
    """Gemma3ForConditionalGeneration"""
    if parameters=='default':
        parameters = '12b'
    if version=='default':
        version = '3'

    parameters = parameters.lower()
    model_name = f"google/gemma-{version}-{parameters}-it"

    llm = LLM(
        model=model_name,
        max_model_len=2048,
        max_num_seqs=2,
        mm_processor_kwargs={"do_pan_and_scan": True},
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        limit_mm_per_prompt={"image": 1},
    )

    processor = AutoProcessor.from_pretrained(model_name)

    return llm, None, processor

def get_internvl(parameters='8B', version='3'):
    """OpenGVLab/InternVL3-8B"""
    if parameters=='default':
        parameters = '8B'
    if version=='default':
        version = '3'

    model_name = f"OpenGVLab/InternVL{version}-{parameters}"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        disable_mm_preprocessor_cache=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
    )

    processor = AutoTokenizer.from_pretrained(model_name)

    #stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    #stop_token_ids = [processor.convert_tokens_to_ids(i) for i in stop_tokens]

    return llm, None, processor

def get_llava_onevision_qwen2(parameters='8B', version='3'):
    """llava-hf/llava-onevision-qwen2-7b-ov-hf"""
    model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    llm = LLM(
        model=model_name,
        max_model_len=16384,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.90,
        disable_mm_preprocessor_cache=True,
    )
    # No special stop tokens by default
    stop_token_ids = None

    processor = AutoTokenizer.from_pretrained(model_name)

    return llm, stop_token_ids, processor

def get_mllama(parameters='11B', version='3.2'):
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    llm = LLM(model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        limit_mm_per_prompt={"image": 1},
    )

    processor = AutoTokenizer.from_pretrained(model_name)

    return llm, None, processor


SAVE_EVERY = 100

MODEL_MAP = {
    'QwenOmni': get_qwen_omni,
    'InternVL': get_internvl,                       # default: 8B, 3
    'Gemma3': get_gemma3,                           # default: 12b, 3
    'Qwen': get_qwen_vl,                            # default: 7B, 2.5
    'LlavaOneVision': get_llava_onevision_qwen2,    # default: 7B, ?
    'MLlama': get_mllama,                           # default: 11B, 3.2
}


def encode_image(image_path):
    size = 512
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width_percent = size / float(img.size[0])
            new_height = int((float(img.size[1]) * width_percent))

            img_resized = img.resize((size, new_height), Image.LANCZOS)

            buffered = BytesIO()
            img_resized.save(buffered, format="PNG")

            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        return None  
    
def build_prompt(model_name, processor, image, context,):

    system_prompt = """
        You are a helpful assistant that verifies whether an image faithfully follows a short descriptive text. Explicitly check if both the individuals in the text are present in the image. The user provides a descriptive sentence and an image. Your task is to evaluate if the image matches the description. Your response must be a JSON object with three keys:

        "reason": a concise explanation of your reasoning,
        "follows": a boolean indicating whether the image follows the description,
        "confidence": a number from 0 to 10 reflecting your confidence in the judgment.
    """

    pre_prompt = """
        Here is a text and an image. Determine whether the image faithfully follows the text. Explicitly check if both the individuals in the text are present in the image. Output your response in this format:
        {"reason": "reasoning", "follows": True or False, "confidence": number from 0 to 10}
    """

    post_prompt = f"Give the output in strict JSON format. \n" + """{\"reason\": \"Explain your reasoning here\", \"follows\": \"True or False\", \"confidence\": \"0 to 10\"}"""

    prompt = pre_prompt + f"\nText: {context}" + post_prompt
    #print(prompt)

    image_asset = Image.open(image).convert("RGB")
    messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
    ]
    

    if model_name.lower() == 'qwenomni':
        default_system = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba " +\
            "Group, capable of perceiving auditory and visual inputs, as well as " +\
            "generating text and speech."
        base64_img = f'data:image;base64,{encode_image(image)}'
        messages = [
                {"role": "system", "content": default_system + '\n' + system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": base64_img, "min_pixels": 224 * 224, "max_pixels": 1280 * 28 * 28},
                        {"type": "text", "text": prompt},
                    ],
                },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        mm_data = {"image": image_inputs} if image_inputs is not None else {}

        llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}

        return llm_inputs

    elif model_name.lower() == 'qwen':
        base64_img = f'data:image;base64,{encode_image(image)}'
        messages = [
                {"role": "system", "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group." + '\n' + system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": base64_img, "min_pixels": 224 * 224, "max_pixels": 1280 * 28 * 28},
                        {"type": "text", "text": prompt},
                    ],
                },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        mm_data = {"image": image_inputs} if image_inputs is not None else {}

        llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}

        return llm_inputs
    
    elif model_name.lower() == 'internvl':
        ## test InternVL3
        messages = [{
                'role': 'user',
                'content': f"<image>\n{prompt}"
            }]
        prompts = processor.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
        
        llm_inputs = { "prompt": prompts,
                "multi_modal_data": {"image": image_asset}
        }

        return llm_inputs
    
    elif model_name.lower() == 'llavaonevision':
        prompt_str = f"<|im_start|>user <image>\n{prompt}<|im_end|>" + \
                            "<|im_start|>assistant\n"
        llm_inputs = { "prompt": prompt_str,
                "multi_modal_data": {"image": image_asset}
        }

        return llm_inputs
    
    elif model_name.lower() == 'mllama':
        llm_inputs = processor.apply_chat_template(messages,
                                            add_generation_prompt=True,
                                            tokenize=False)
        return llm_inputs
    
    elif model_name.lower() == 'gemma3':
        prompt_str = ("<bos><start_of_turn>user\n"
                f"<start_of_image>{prompt}<end_of_turn>\n"
                "<start_of_turn>model\n")
        
        llm_inputs = { "prompt": prompt_str,
                "multi_modal_data": {"image": image_asset}
        }

        return llm_inputs
        
    else:
        raise NotImplementedError

def run_inference(model, prompt, sampling_params):
    generated = model.generate([prompt], sampling_params=sampling_params)
    return generated[0].outputs[0].text


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

def main(args):
    model_name = args.model_name
    version = args.version
    parameters = args.parameters

    build_llm_fn = MODEL_MAP[model_name]
    model, stop_token_ids, processor = build_llm_fn(parameters=parameters, version=version)

    sampling_params = SamplingParams(
        temperature=0.0,        # Deterministic
        max_tokens=1024,
        stop_token_ids=stop_token_ids,
    )

    df = pd.read_csv(args.csv_path)
    if 'question_polarity' in df.columns:
        df = df[df['question_polarity'] == 'nonneg']
    total = len(df)
    chunk_size = total // args.split_in
    start = (args.chunk - 1) * chunk_size
    end = start + chunk_size if args.chunk < args.split_in else total
    df = df.iloc[start:end].reset_index(drop=True)
    failed_records = []

    # ----------------------------------------------------------------
    base_dir = args.base

    # 1a. build the path
    df['folder_path'] = df.apply(
        lambda row: os.path.join(
            base_dir,
            row.get('category', ''),
            row.get('unique_question_id', '')
        ),
        axis=1
    )

    # 1b. keep only rows whose folder actually exists
    df = df[df['folder_path'].apply(os.path.isdir)].reset_index(drop=True)

    # 2a. prepare the 50 column names
    image_cols = [f'image_{i+1}' for i in range(50)]
    # initialize them all to None
    for col in image_cols:
        df[col] = None

    # 2b. fill in the actual image paths (up to 50 per folder)
    for idx, folder in df['folder_path'].items():
        # list & sort image files in that folder
        files = sorted([
            f for f in os.listdir(folder)
            if (
                os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith(IMAGE_EXTS)
            )
        ])
        
        # assign each to the corresponding column
        for i, fname in enumerate(files):
            df.at[idx, f'image_{i+1}'] = os.path.join(folder, fname)

    # 3a. melt the wide form into long form
    id_vars = [c for c in df.columns if c not in image_cols]
    df = df.melt(
        id_vars=id_vars,
        value_vars=image_cols,
        var_name='image_num',
        value_name='image_path'
    )

    # 3b. drop the “empty” image slots
    df = df.dropna(subset=['image_path']).reset_index(drop=True)

    batch_inputs = []
    batch_record_map = []  # Keep track of which row & image_col => index in outputs
    results_data = []

    batch_processed = 0

    print(df.shape)
    
    for row_idx, row in df.iterrows():
        image_path = row.get('image_path')
        uid = row.get('uid')
        unique_question_id = row.get('unique_question_id')
        simplified_context = row.get('simplified_context')
        context = row.get('context')

        try:
            prompt = build_prompt(model_name, processor, image_path, simplified_context,)
        except Exception as e:
            failed_records.append({
                "row_idx": row_idx,
                "uid": uid,
                "unique_question_id": unique_question_id,
                "image_path": image_path,
                "error": str(e)
            })
            continue

        batch_inputs.append(prompt)
        batch_record_map.append((row_idx, uid, unique_question_id, image_path, context,))

        if ((((row_idx+1) % args.batch_size) == 0) or (row_idx==df.shape[0]-1)):
            print(f"Running inference on {len(batch_inputs)} items...")
            outputs = model.generate(batch_inputs, sampling_params=sampling_params)
            batch_processed += 1

            for out_idx, output_obj in enumerate(outputs):
                text_output = output_obj.outputs[0].text

                row_idx, uid, unique_question_id, image_path, context = batch_record_map[out_idx]

                results_data.append({
                            "row_idx": row_idx,
                            "uid": uid , 
                            "unique_question_id": unique_question_id, 
                            "image_path": image_path,
                            "context": context,
                            "model_output": text_output
                        })
            
            df_results = pd.DataFrame(results_data)

            output_path = args.output_path
            os.makedirs(output_path, exist_ok=True)
            print(f">>> Inference partially done: {len(batch_inputs)}. Saving results to {output_path}")
            # import pdb; pdb.set_trace()
            df_results.to_csv(os.path.join(output_path, f'results_chunk{args.chunk}.csv'), index=False)

            # Clear for next batch
            batch_inputs = []
            batch_record_map = []

    print(f">> Inference done. Results saved to {output_path}")
    # write out any failures for this chunk
    failed_path = os.path.join(output_path, f"failed_chunk{args.chunk}.json")
    with open(failed_path, "w") as f:
        import json
        json.dump(failed_records, f, indent=2)
    print(f">> Recorded {len(failed_records)} failures to {failed_path}")



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
    ## Usage: python run_inference_batch.py --model_name InternVL --csv_path "/home/ja339952/CAP6412/temp_context.csv" --batch_size 10
    parser = argparse.ArgumentParser(description="Batch inference for specific V+L models.")
    parser.add_argument("--model_name", type=str, required=False, default="Qwen",
                        help="Which model to run. Must be one of: " + ", ".join(MODEL_MAP.keys()))
    parser.add_argument("--version", type=str, default="default",
                        help="Version 2/2.5/3/...")
    parser.add_argument("--parameters", type=str, default="default",
                        help="Parameter count 3b, 4b, 7B, 8B, ...")
    parser.add_argument("--csv_path", type=str, default="./",
                        help="File path to the dataset.")
    parser.add_argument('--base', default='images/scraped',
                        help='Image directory base path')
    parser.add_argument("--output_path", type=str, default="lmm_results",
                        help="File path to store inference outputs CSV.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=1,
                        help="Out of split_n chunks.")
    parser.add_argument("--split_in", type=int, default=10,
                        help="Split the dataset into n chunks for parallel job submission.")

    args = parser.parse_args()

    check_pytorch_gpu()
    main(args)
