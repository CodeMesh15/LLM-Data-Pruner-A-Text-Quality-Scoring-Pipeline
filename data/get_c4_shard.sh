
# This script downloads a small shard of the C4 (Colossal Cleaned Common Crawl) dataset.
# It uses the Hugging Face `datasets` library to stream the data, avoiding a full download.

# Create the data directory if it doesn't exist
mkdir -p data

echo "--- Downloading a 10,000-sample shard of the C4 dataset ---"
echo "This may take a few minutes depending on your connection..."

# Use a Python one-liner to download and format the data
python -c "
from datasets import load_dataset
import json
from tqdm import tqdm

# Load the C4 dataset in streaming mode to avoid downloading everything
dataset = load_dataset('c4', 'en', split='train', streaming=True)

# Take the first 10,000 samples from the stream
num_samples = 10000
shard = dataset.take(num_samples)

output_file = 'data/c4_shard.jsonl'

print(f'Writing {num_samples} samples to {output_file}...')

# Write the samples to a JSONL file
with open(output_file, 'w', encoding='utf-8') as f:
    for item in tqdm(shard, total=num_samples):
        # We only need the 'text' field for this project
        line = json.dumps({'text': item['text']})
        f.write(line + '\n')

print('--- Download complete! ---')
"
