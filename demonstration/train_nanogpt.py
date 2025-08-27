
import os
import subprocess
import sys

# --- IMPORTANT ---
# This script is a WRAPPER and assumes you have a nanoGPT-style
# training environment set up. You must clone nanoGPT or a similar
# repository for this to work.
#
# 1. Clone nanoGPT: `git clone https://github.com/karpathy/nanoGPT.git`
# 2. Place this script in the root of the nanoGPT directory or adjust paths.

NANO_GPT_DIR = "nanoGPT" # Adjust if you cloned it elsewhere

def run_training_experiment(dataset_name, data_path, num_iters=2000):
    """
    Automates the process of preparing data and running a nanoGPT training session.
    """
    print(f"\n--- Starting Experiment for: {dataset_name} ---")
    
    # --- Step 1: Prepare the data (binarize it for nanoGPT) ---
    prepare_script = os.path.join(NANO_GPT_DIR, "data/shakespeare/prepare.py") # Use any of their prepare scripts as a template
    if not os.path.exists(prepare_script):
        print(f"Error: `prepare.py` not found in nanoGPT directory. Cannot proceed.")
        return

    print(f"Preparing data from {data_path}...")
    # This command will tokenize the text and create train.bin and val.bin
    # For a real run, you'd need to adapt the prepare script to read your text file.
    # For this demo, we assume the data is already in a simple text format.
    # subprocess.run([sys.executable, prepare_script, "--input_file", data_path, "--output_dir", f"data/{dataset_name}"])

    # --- Step 2: Run the training script ---
    train_script = os.path.join(NANO_GPT_DIR, "train.py")
    if not os.path.exists(train_script):
        print(f"Error: `train.py` not found in nanoGPT directory. Cannot proceed.")
        return

    print(f"Starting training on {dataset_name} data for {num_iters} iterations...")
    # This command runs the training. Logs will be printed to the console.
    # We are training a very small model for demonstration purposes.
    command = [
        sys.executable, train_script,
        # f"--dataset=data/{dataset_name}", # This would point to the binarized data
        "--compile=False",
        "--eval_interval=100",
        "--block_size=64",
        "--batch_size=8",
        "--n_layer=4",
        "--n_head=4",
        "--n_embd=128",
        "--max_iters=" + str(num_iters),
        "--lr_decay_iters=" + str(num_iters),
        "--dropout=0.0"
    ]
    
    print(f"Running command: {' '.join(command)}")
    print("\n--- MOCK RUN ---")
    print("In a real run, the nanoGPT training would start now.")
    print(f"Observe the final validation loss for the '{dataset_name}' model.")
    # In a real scenario, you would uncomment the line below:
    # subprocess.run(command)

def main():
    """
    Runs two training experiments: one on original data, one on pruned data.
    """
    original_data_path = "data/c4_shard.jsonl" # Assume this exists
    pruned_data_path = "data/c4_shard_pruned.jsonl"   # Assume you created this with pipeline.py

    if not os.path.exists(NANO_GPT_DIR):
        print("Error: nanoGPT repository not found. Please clone it first.")
        print("`git clone https://github.com/karpathy/nanoGPT.git`")
        return
        
    print("This script will now run two mock training sessions.")
    print("The goal is to compare the final validation loss.")
    print("The model trained on the 'pruned' data is expected to achieve a lower loss,")
    print("demonstrating the effectiveness of the data quality pipeline.")

    # Run for original data
    run_training_experiment(
        dataset_name="original_data",
        data_path=original_data_path
    )
    
    # Run for pruned data
    run_training_experiment(
        dataset_name="pruned_data",
        data_path=pruned_data_path
    )

if __name__ == '__main__':
    main()
