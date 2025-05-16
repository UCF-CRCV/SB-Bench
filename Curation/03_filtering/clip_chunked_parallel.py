#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_row(args):
    idx, row, model, processor, device, base_path = args
    text = row.get('simplified_context', '')
    path = os.path.join(
        base_path,
        row.get('category', ''), row.get('unique_question_id', '')
    )
    # print(path)
    # collect image files
    if os.path.isdir(path):
        imgs = [os.path.join(path, f) for f in os.listdir(path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    elif os.path.isfile(path):
        imgs = [path]
    else:
        return []
    if not imgs:
        return []
    # load & preprocess once per row
    pil_imgs = []
    for p in imgs:
        try:
            pil_imgs.append(Image.open(p).convert('RGB'))
        except OSError as e:
            print(f"Warning: skipping truncated image {p}: {e}")
    if not pil_imgs:
        return []
    
    inputs = processor(text=[text] * len(pil_imgs), images=pil_imgs,
                       return_tensors='pt', padding=True)
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    img_embeds = outputs.image_embeds
    txt_embeds = outputs.text_embeds
    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
    txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)
    sims = (img_embeds * txt_embeds).sum(dim=-1).cpu().tolist()
    return list(zip(imgs, sims))


def main():
    parser = argparse.ArgumentParser(
        description="Parallel CLIP similarity scoring with progress bar")
    parser.add_argument('--csv', required=True, help='Path to input CSV')
    parser.add_argument('--chunk', type=int, choices=range(1,5), required=True,
                        help='Which of the 4 chunks to process (1-4)')
    parser.add_argument('--model_name',
                        default='openai/clip-vit-large-patch14',
                        help='HuggingFace model ID for CLIP')
    parser.add_argument('--base_path', default='images/scraped',
                        help='Image directory base path')
    parser.add_argument('--output', default=None,
                        help='Output CSV filepath')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='Save checkpoint after this many images')
    parser.add_argument('--workers', type=int,
                        default=os.cpu_count(),
                        help='Number of parallel workers')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # print(df.shape)
    if 'question_polarity' in df.columns:
        df = df[df['question_polarity'] == 'nonneg']
    total = len(df)
    chunk_size = total // 4
    start = (args.chunk - 1) * chunk_size
    end = start + chunk_size if args.chunk < 4 else total
    df_chunk = df.iloc[start:end].reset_index(drop=True)
    # print(df_chunk.shape)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # warm up GPU/model once
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()
    # optional light warm-up: do a single dummy forward to prime kernels
    try:
        dummy_text = [""]  # empty string
        dummy_img = Image.new("RGB", (224,224), color=(0,0,0))
        inputs = processor(text=dummy_text, images=[dummy_img], return_tensors="pt", padding=True)
        inputs = {k: v.to(device, non_blocking=True) for k,v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
    except Exception:
        pass
    
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        out_file = os.path.join(args.output, f"results_chunk{args.chunk}.csv")
    else:
        os.makedirs('clip_results', exist_ok=True)
        out_file = f"clip_results/results_chunk{args.chunk}.csv"
        
    results = []
    processed = 0

    # Prepare all tasks upfront
    tasks = [(idx, row, model, processor, device, args.base_path) for idx, row in df_chunk.iterrows()]

    # Use ThreadPoolExecutor + tqdm
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # as_completed wrapped in tqdm to show per-row progress
        for res in tqdm(executor.map(process_row, tasks),
                        total=len(tasks),
                        desc="Rows processed"):
            for img_path, sim in res:
                results.append({'image_path': img_path,
                                'unique_question_id': img_path.split('/')[-2],
                                'category': img_path.split('/')[-3],
                                'similarity_score': sim})
                processed += 1
                if processed % args.save_every == 0:
                    # print(results)
                    pd.DataFrame(results).to_csv(out_file, index=False)

    # final save
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"Done. Processed {processed} images. Results saved to {out_file}.")


if __name__ == '__main__':
    main()
