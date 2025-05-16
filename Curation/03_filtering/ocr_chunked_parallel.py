import os
import argparse
import json
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing as mp
from paddleocr import PaddleOCR
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import torch  # for GPU check

# ========== Config ==========
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FLAGS_use_cuda"] = "0"
# Limit MKL/OpenBLAS threads per process to prevent oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Will be initialized per‐process
reader = None

def init_worker(use_gpu: bool):
    global reader
    os.environ.setdefault('FLAGS_allocator_strategy', 'auto_growth')
    reader = PaddleOCR(use_angle_cls=True, lang='en', gpu=use_gpu, show_log=False)

def process_item(task):
    """OCR all images under one (category, uid).  Never raises."""
    try:
        category, uid, base = task
        path = os.path.join(base, category or '', uid or '')

        # gather image paths
        if os.path.isdir(path):
            imgs = [os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.gif'))]
        elif os.path.isfile(path):
            imgs = [path]
        else:
            # should be filtered out already
            return []

        out = []
        for img_path in imgs:
            try:
                detections = reader.ocr(img_path, cls=True)
                words, bboxes, confidences = [], [], []

                if detections:
                    for line in detections:
                        if line:
                            for word_info in line:
                                if word_info:
                                    box, (text, conf) = word_info
                                    words.append(text)
                                    bboxes.append(box)
                                    confidences.append(conf)

                # apply filters and dedupe
                filtered = []
                seen = set()
                for w in words:
                    lw = w.lower()
                    # drop single characters
                    if len(w) == 1:
                        continue
                    # drop URLs and domains
                    if lw.startswith('www.') or lw.endswith('.com'):
                        continue
                    # dedupe
                    if w not in seen:
                        filtered.append(w)
                        seen.add(w)
                
                out.append({
                    'image_path': img_path,
                    'words': ' '.join(words), 
                    'filtered_words': ' '.join(filtered),
                    'ocr_rejected': len(filtered)>3, 
                    'bboxes': bboxes,
                    'confidences': confidences,
                })
            except Exception as e_img:
                out.append({
                    'image_path': img_path,
                    'words': f"[IMG ERROR: {e_img}]", 
                    'ocr_rejected': None, 
                    'bboxes': None,
                    'confidences': None,
                })
        return out

    except Exception as e:
        # never let a crash bubble up
        return [{
            'image_path': None,
            'left': None, 'top': None,
            'width': None,'height': None,
            'text': f"[TASK ERROR: {e}]",
            'confidence': None
        }]

def main():
    parser = argparse.ArgumentParser(
        description="Chunked, parallel OCR via PaddleOCR (robust to crashes)")
    parser.add_argument('--csv',       required=True, help='Input CSV')
    parser.add_argument('--chunk',     type=int, choices=range(1,5), required=True,
                        help='Which of the 4 chunks to process (1-4)')
    parser.add_argument('--output',    default=None,
                        help='Output directory (default: ocr_results/)')
    parser.add_argument('--base', default='images/scraped',
                        help='Image directory base path')
    parser.add_argument('--save_every',type=int, default=5000,
                        help='Checkpoint every N entries')
    parser.add_argument('--workers',   type=int, default=min(4, os.cpu_count()//2),
                        help='Number of processes')
    args = parser.parse_args()

    # 1) Load & chunk your CSV
    df = pd.read_csv(args.csv)
    if 'question_polarity' in df.columns:
        df = df[df['question_polarity']=='nonneg']
    total   = len(df)
    chunk_sz= total // 4
    start   = (args.chunk-1)*chunk_sz
    end     = start+chunk_sz if args.chunk<4 else total
    dfc     = df.iloc[start:end].reset_index(drop=True)

    print(f"Processing {len(dfc)} rows (chunk {args.chunk}/4)")

    # 2) Prepare output paths
    out_dir      = args.output or 'ocr_results'
    os.makedirs(out_dir, exist_ok=True)
    out_csv      = os.path.join(out_dir, f"results_chunk{args.chunk}.csv")
    missing_json = os.path.join(out_dir, f"missing_chunk{args.chunk}.json")

    # 3) Detect missing and build tasks
    base    = args.base
    tasks   = []
    missing = []
    for _, row in dfc.iterrows():
        cat = row.get('category','')
        uid = row.get('unique_question_id','')
        pth = os.path.join(base, cat, uid)
        if not (os.path.isdir(pth) or os.path.isfile(pth)):
            missing.append({'category': cat, 'unique_question_id': uid, 'path': pth})
        else:
            tasks.append((cat, uid, base))

    # save missing
    with open(missing_json, 'w') as f:
        json.dump(missing, f, indent=2)
    print(f"Found {len(missing)} missing; wrote {missing_json}")

    # 4) Parallel OCR, catching per‐future errors
    use_gpu = torch.cuda.is_available()
    ctx     = mp.get_context('forkserver')
    all_results = []
    processed   = 0

    with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=init_worker,
            initargs=(use_gpu,)
        ) as executor:

        futures = { executor.submit(process_item, t): t for t in tasks }
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Rows OCR'ed"):
            task_info = futures[future]
            try:
                res = future.result()
            except Exception as e_fut:
                print(f"[FUTURE ERROR] task={task_info}: {e_fut}")
                continue

            for entry in res:
                all_results.append(entry)
                processed += 1
                if processed % args.save_every == 0:
                    pd.DataFrame(all_results).to_csv(out_csv, index=False)

    # 5) Final save
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"Done. Extracted {processed} entries. Saved to {out_csv}")

if __name__ == '__main__':
    main()