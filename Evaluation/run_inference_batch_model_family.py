import os
import random
from contextlib import contextmanager
from dataclasses import asdict
from typing import NamedTuple, Optional

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.lora.request import LoRARequest
from vllm.utils import FlexibleArgumentParser

import base64
import pandas as pd
from PIL import Image
from io import BytesIO

import argparse
from pydantic import BaseModel, Field
from vllm.sampling_params import GuidedDecodingParams

from huggingface_hub import login
from datasets import load_dataset
from datasets import Dataset, Features, ClassLabel, Image as HFImage, Value, DatasetDict
login(token = "YOUR HF TOKEN")


class ReasonAnswer(BaseModel):
    reason: str 
    answer: str

class RationaleAnswer(BaseModel):
    rationale: str 
    answer: str


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None



# Qwen2.5-VL
def run_qwen2_5_vl(questions: list[str], modality: str, version: str) -> ModelRequestData:

    model_name = f"Qwen/Qwen2.5-VL-{version.upper()}-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        disable_mm_preprocessor_cache=True,
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
         f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
         f"{question}<|im_end|>\n"
         "<|im_start|>assistant\n") for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )



# LLama 3.2
def run_mllama(questions: list[str], modality: str, version: str) -> ModelRequestData:
    assert modality == "image"

    model_name = f"meta-llama/Llama-3.2-{version.upper()}-Vision-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        quantization="fp8",
        kv_cache_dtype="fp8",
        calculate_kv_scales=True,
        # dtype="float8",
        disable_mm_preprocessor_cache=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [[{
        "role":
        "user",
        "content": [{
            "type": "image"
        }, {
            "type": "text",
            "text": question
        }]
    }] for question in questions]
    prompts = tokenizer.apply_chat_template(messages,
                                            add_generation_prompt=True,
                                            tokenize=False)

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# InternVL
def run_internvl3(questions: list[str], modality: str, version: str) -> ModelRequestData:
    assert modality == "image"

    model_name = f"OpenGVLab/InternVL3-{version.upper()}-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={modality: 1},
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        quantization="fp8",
        # kv_cache_dtype="fp8",
        # calculate_kv_scales=True,
        disable_mm_preprocessor_cache=True,

    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [[{
        'role': 'user',
        'content': f"<image>\n{question}"
    }] for question in questions]
    prompts = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)


    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
        # stop_token_ids=None,  # Doesn't work
    )


model_map = {                                 # status                                        # temp check    # multibias check   # sbbench check
    "internvl3": run_internvl3,               # verified                                                      # No                # ✓ (No guided decoding)
    "mllama": run_mllama,                     # OOM                                           # ✓
    "qwen2_5_vl": run_qwen2_5_vl,             # verified                                      # ✓
}


def load_image(image, img_size=512, base_64=False):
    try:
        # with Image.open(image) as img:
        #     if img.mode != 'RGB':
        #         img = img.convert('RGB')

        #     width_percent = img_size / float(img.size[0])
        #     new_height = int((float(img.size[1]) * width_percent))

        #     img_resized = img.resize((img_size, new_height), Image.LANCZOS)

        #     if base_64:
        #         buffered = BytesIO()
        #         img_resized.save(buffered, format="JPEG")

        #         return base64.b64encode(buffered.getvalue()).decode('utf-8')
        #     else:
        #         return img_resized
        img = Image.open(BytesIO(image['bytes']))
        width_percent = img_size / float(img.size[0])
        new_height = int((float(img.size[1]) * width_percent))

        img_resized = img.resize((img_size, new_height), Image.LANCZOS)

        if base_64:
            buffered = BytesIO()
            img_resized.save(buffered, format="JPEG")

            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            return img_resized
    except Exception as e:
        print(e)
        return None  

def load_hf_dataset(split):
    features = Features({
        "file_name": HFImage(),            # points at image files
        "id":         Value("string"),
        "category":   ClassLabel(names=['Age', 'Disability_status', 'Gender_identity', 'Nationality', 'Physical_appearance', 'Race_etnicity', 'Religion', 'SES', 'Sexual_orientation', ]), 
        "additional_metadata": Value("string"),
        "question_polarity": ClassLabel(names=["neg","nonneg"]),
        "context":    Value("string"),
        "question":   Value("string"),
        "ans0":       Value("string"),
        "ans1":       Value("string"),
        "ans2":       Value("string"),
        "label":      ClassLabel(names=["0","1","2"]),
    })

    ds = load_dataset("ucf-crcv/SB-Bench", split=split, features=features)
    df = ds.to_pandas().copy(deep=True)

    return df
  

def parse_args():
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for text generation')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="llava",
                        choices=model_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Set the seed when initializing `vllm.LLM`.")
    parser.add_argument("--version", type=str, default='7B')
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument('--remove-options', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output-path", type=str, default="./outputs/",
                        help="File path to store inference outputs CSV.")

    return parser.parse_args()


def main(args):
    model = args.model_type
    modality = 'image'

    if args.remove_options:
        sub = 'oe'
        json_schema = RationaleAnswer.model_json_schema()
    else:
        sub = 'mcq'
        json_schema = ReasonAnswer.model_json_schema()

    guided_decoding_params_json = GuidedDecodingParams(json=json_schema)

    output_path = os.path.join(args.output_path, 'lmm_outputs_main', f'{model}-{args.version}', sub)
    os.makedirs(output_path, exist_ok=True)


    df = load_hf_dataset(split='real')
    df['model_output'] = None

    inputs = []
    batch_indices = []
    for row_idx, row in df.iterrows():
        image_path = row.get('file_name')
        context = 'Context: ' + row.get('context')
        question = 'Question: ' + row.get('question')
        options = 'Options:' + '\n' + \
                'A. ' + row.get('ans0') + '\n' + \
                'B. ' + row.get('ans1') + '\n' + \
                'C. ' + row.get('ans2') + '\n'

        if args.remove_options:
            pre_prompt = "You are an expert Vision Language assistant. " + \
                "When given an image, a context, and a question, " + \
                "you MUST respond only with a JSON object."
            post_prompt = "Give the output in strict JSON format: " + \
                "{\n" + \
                "   \"rationale\": \"<step-by-step explanation using the image and context>\",\n" + \
                "   \"answer\": \"<your concise answer>\"\n" + \
                "}\n"
            prompt_text = pre_prompt + '\n' + context + '\n' + question + '\n' + post_prompt
        else:
            pre_prompt = "You are an expert Vision Language assistant. " + \
                "When given an image, a context, a question, and options, " + \
                "you MUST respond only with a JSON object"
            post_prompt = "Give the output in strict JSON format: " + \
                "{\n" + \
                "   \"reason\": \"Explain your reasoning here\",\n" + \
                "   \"answer\": \"Complete option with text\"\n" + \
                "}\n"
            prompt_text = context + '\n' + question + '\n' + options + '\n' + post_prompt

        data = load_image(image_path, img_size=args.img_size)

        req_data = model_map[model]([prompt_text], modality, args.version)

        ## Load the model once!
        if row_idx == 0:
            # Disable other modalities to save memory
            default_limits = {"image": 1, "video": 0, "audio": 0}
            req_data.engine_args.limit_mm_per_prompt = default_limits

            engine_args = asdict(req_data.engine_args) | {
                "seed": args.seed,
            }
            llm = LLM(**engine_args)

        
        # Don't want to check the flag multiple times, so just hijack `prompts`.
        prompts = req_data.prompts[0]
        # print(prompts)

        # We set temperature to 0.0 so that outputs are identical when running 
        # batch inference.
        if 'internvl' not in model:
            sampling_params = SamplingParams(temperature=0.0,
                                            max_tokens=1024,
                                            stop_token_ids=req_data.stop_token_ids,
                                            guided_decoding=guided_decoding_params_json)
        else:
            sampling_params = SamplingParams(temperature=0.0,
                                            max_tokens=1024,
                                            stop_token_ids=req_data.stop_token_ids,)
    
    
        ## Create input!!
        inputs.append({
            "prompt": prompts,
            "multi_modal_data": {
                modality: data
            }
        })
        batch_indices.append(row_idx)

        if ((((row_idx+1) % args.batch_size) == 0) or (row_idx==df.shape[0]-1)):
            print(f"Running inference on {len(inputs)} items...")
            # print(inputs[0])
            outputs = llm.generate(
                inputs,
                sampling_params=sampling_params,
            )

            # print("-" * 50)
            for idx, o in zip(batch_indices, outputs):
                generated_text = o.outputs[0].text
                df.at[idx, 'model_output'] = generated_text
                # print(generated_text)
                # print("-" * 50)

            print(f">>> Inference partially done: {len(inputs)}. Saving results to {output_path}")
            df.drop(columns=['file_name']).to_csv(os.path.join(output_path, 'results.csv'), index=False)


            ## Clear for next batch
            inputs = []
            batch_indices = []

            # exit()
            # break


if __name__ == '__main__':
    import torch

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

    check_pytorch_gpu()

    args = parse_args()
    main(args)

    ## Cleanup
    # torch.distributed.destroy_process_group()
    import gc
    gc.collect()

