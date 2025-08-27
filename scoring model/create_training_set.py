

import pandas as pd
from datasets import load_dataset
import json
from tqdm import tqdm
import os

def create_training_set(num_samples_per_class=5000, output_dir='scoring_model'):
    """
    Creates a labeled dataset with high-quality (Wikipedia) and
    low-quality (C4) text samples.
    """
    print("--- Creating Training Set for Quality Scorer ---")
    
    # --- 1. Fetch High-Quality Data (Wikipedia) ---
    print(f"Fetching {num_samples_per_class} high-quality samples from Wikipedia...")
    wiki_dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming=True)
    wiki_samples = list(wiki_dataset.take(num_samples_per_class))
    
    high_quality_df = pd.DataFrame(wiki_samples)
    high_quality_df = high_quality_df[['text']] # Keep only the text column
    high_quality_df['label'] = 1 # 1 for high quality
    
    # --- 2. Fetch Low-Quality Data (C4) ---
    print(f"Fetching {num_samples_per_class} low-quality samples from C4...")
    c4_shard_path = 'data/c4_shard.jsonl'
    if not os.path.exists(c4_shard_path):
        print(f"Error: '{c4_shard_path}' not found. Please run get_c4_shard.sh first.")
        return

    low_quality_samples = []
    with open(c4_shard_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples_per_class:
                break
            low_quality_samples.append(json.loads(line))
            
    low_quality_df = pd.DataFrame(low_quality_samples)
    low_quality_df['label'] = 0 # 0 for low quality

    # --- 3. Combine and Save ---
    print("Combining and shuffling the dataset...")
    combined_df = pd.concat([high_quality_df, low_quality_df], ignore_index=True)
    
    # Shuffle the dataset
    shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'quality_training_set.csv')
    shuffled_df.to_csv(output_path, index=False)
    
    print(f"--- Training set created successfully at '{output_path}' ---")
    print(f"Total samples: {len(shuffled_df)}")


if __name__ == '__main__':
    create_training_set()
