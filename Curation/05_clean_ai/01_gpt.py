import os
import json
import base64
from PIL import Image
from io import BytesIO
import openai
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import os
from openai import AzureOpenAI

endpoint = os.getenv("ENDPOINT_URL", "YOUR AZURE ENDPOINT URL")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "YOUR AZURE OPENAI KEY")


# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-10-21",
)

def encode_image(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width_percent = 512 / float(img.size[0])
            new_height = int((float(img.size[1]) * width_percent))

            img_resized = img.resize((512, new_height), Image.LANCZOS)

            buffered = BytesIO()
            img_resized.save(buffered, format="JPEG")

            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        return None  # Handle broken images gracefully


# ========== New helper to process one row ==========
def process_row(row, deployment):
    """
    Takes a dict-like row with keys ['unique_image_id','image_path','simplified_context']
    and returns either a batch-entry dict or None (if image encoding failed).
    """
    uid = row['id']
    b64 = encode_image(row['file_name'])
    if b64 is None:
        return None

    system_prompt = (
        "You are an image classification assistant that determines if an image is cartoonistic or AI-generated. "
        "Analyze the given image and provide a response with the following fields: reason, confidence (0-10), and answer (True/False). " 
        "Respond strictly in JSON format."
    )

    prompt_eval = "Analyze the following image and provide the output in JSON format:\n\nRESPONSE FORMAT:\n{\n  \"reason\": \"give your reasoning here\",\n  \"confidence\": 0-10,\n  \"answer\": true/false\n}"
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_eval},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        },
        {"role": "system", "content": system_prompt}
    ]

    return {
        "custom_id": str(uid),
        "method": "POST",
        "url": "/chat/completions",
        "body": {
            "model": deployment,
            "messages": PROMPT_MESSAGES,
            "max_tokens": 1000,
            "temperature": 0.4
        }
    }



# ========== Parallel generate_caption ==========
def generate_caption(labels_csv, deployment, batch_size=100, max_workers=None):
    df = pd.read_csv(labels_csv)
    df = df[df['question_polarity']=='nonneg']
    records = df.to_dict(orient="records")

    dl = []         # successful batch entries
    failed = []     # list of IDs for which encoding failed

    # Use os.cpu_count() - 1 by default, or override via max_workers
    workers = max_workers or max(1, os.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_row, row, deployment): row['id']
            for row in records
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Encoding & building prompts"):
            uid = futures[fut]
            bat = fut.result()
            if bat is None:
                failed.append(uid)
            else:
                dl.append(bat)

    invalid_image_count = len(failed)
    print(f"\nFinished encoding: {len(dl)} valid images, {invalid_image_count} failed.")

    # … after you've collected `dl` (the list of batch‐entries) …

    out_dir = "batchFiles"
    os.makedirs(out_dir, exist_ok=True)

    # 180 MB threshold
    THRESHOLD = 180 * 1000 * 1000  # bytes: OpenAI considers 1000 instead of 1024!

    batch_idx = 0
    current_size = 0
    current_lines = []

    for entry in dl:
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        size = len(line.encode("utf-8"))

        # if adding this line would exceed threshold, flush current batch
        if current_size + size > THRESHOLD and current_lines:
            path = os.path.join(out_dir, f"batch_{batch_idx}.jsonl")
            with open(path, "w") as f:
                f.writelines(current_lines)
            print(f"Wrote {len(current_lines)} entries to {path} (~{current_size/1000**2:.1f} MB)")
            batch_idx += 1
            current_lines = []
            current_size = 0

        # add line to batch
        current_lines.append(line)
        current_size += size

    # flush any remaining lines
    if current_lines:
        path = os.path.join(out_dir, f"batch_{batch_idx}.jsonl")
        with open(path, "w") as f:
            f.writelines(current_lines)
        print(f"Wrote {len(current_lines)} entries to {path} (~{current_size/1000**2:.1f} MB)")

    print(f"\n===========================\nDone making {batch_idx+1} batch files in {out_dir}!")

if __name__ == "__main__":
    labels_csv = "hf_valid_v2.csv"

    generate_caption(labels_csv, deployment, batch_size=400)
