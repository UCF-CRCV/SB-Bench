import os
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from ImageScraper import download  # Ensure this is correctly installed or available locally

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ResourceWarning)


def download_query(row, limit=50):
    query = row['simplified_context']
    directory = os.path.join('images', 'scraped', row['category'],)
    sub_directory = row['unique_question_id']

    try:
        download(
            query=query,
            sites='',
            license_type="reuse",
            limit=limit,
            directory=directory,
            directory_name=sub_directory,
            min_height=480,
            min_width=640,
        )
        print(f"Downloaded: {query}")
    except Exception as e:
        print(f"Error downloading {query}: {e}")


if __name__ == '__main__':
    print('*' * 100)
    print(f"CPU Count: {os.cpu_count()}")

    # Load CSV
    df = pd.read_csv('AmbiguousBBQsimplifiedQid.csv')

    # Filter only non-negative polarity questions
    df_nonneg = df[df['question_polarity'] == 'nonneg']

    print(df_nonneg.shape[0])


    # Ensure base image directory exists
    base_path = os.path.join('../images/scraped')
    os.makedirs(base_path, exist_ok=True)

    # Parallel download using ThreadPoolExecutor
    max_threads = min(os.cpu_count(), len(df_nonneg))
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(download_query, row) for _, row in df_nonneg.iterrows()]
        for future in as_completed(futures):
            pass  # Placeholder to handle results or errors

    print(f"All downloads completed for: {os.path.basename(base_path)}")
