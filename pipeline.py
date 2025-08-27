
import json
import joblib
import pandas as pd
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch

from feature_engineering.text_metrics import compute_all_metrics

def run_pipeline(args):
    """
    Runs the full data pruning pipeline on a large dataset file.
    """
    print("--- Starting Data Pruning Pipeline ---")
    
    # --- 1. Load all necessary models ---
    print("Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models for feature calculation
    perp_model = AutoModelForCausalLM.from_pretrained(args.perplexity_model).to(device)
    perp_tok = AutoTokenizer.from_pretrained(args.perplexity_model)
    tox_model = AutoModelForSequenceClassification.from_pretrained(args.toxicity_model).to(device)
    tox_tok = AutoTokenizer.from_pretrained(args.toxicity_model)
    
    models = {
        'perplexity_model': perp_model, 'perplexity_tokenizer': perp_tok,
        'toxicity_model': tox_model, 'toxicity_tokenizer': tox_tok,
        'device': device
    }
    
    # Load the trained quality scoring model
    try:
        scoring_model = joblib.load(args.scoring_model_path)
        print("Scoring model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Scoring model not found at '{args.scoring_model_path}'. Please run train_scorer.py first.")
        return

    # --- 2. Process the input file ---
    print(f"Processing input file: {args.input_file}")
    
    lines_processed = 0
    lines_kept = 0
    
    with open(args.input_file, 'r') as infile, open(args.output_file, 'w') as outfile:
        for line in tqdm(infile, desc="Filtering documents"):
            try:
                # Assuming the input is a JSONL file with a 'text' field, like C4
                data = json.loads(line)
                text = data.get('text', '')
                
                if not text or len(text.split()) < 20: # Skip very short texts
                    continue
                
                # a. Calculate features
                metrics = compute_all_metrics(text, models)
                features_df = pd.DataFrame([metrics])
                
                # b. Predict quality score
                # The model predicts probabilities for [low_quality, high_quality]
                quality_prob = scoring_model.predict_proba(features_df)[0][1]
                
                # c. Apply filter
                if quality_prob >= args.threshold:
                    # Write the original JSON line to the output file
                    outfile.write(line)
                    lines_kept += 1
                    
                lines_processed += 1

            except json.JSONDecodeError:
                continue # Skip malformed lines
            except Exception as e:
                print(f"An error occurred on a line: {e}")

    print("\n--- Pipeline Complete ---")
    print(f"Total documents processed: {lines_processed}")
    print(f"Documents kept: {lines_kept}")
    print(f"Pruning ratio: {(lines_processed - lines_kept) / lines_processed:.2%}")
    print(f"Cleaned data saved to: {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the LLM data pruning pipeline.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSONL data file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the filtered output.")
    parser.add_argument('--scoring_model_path', type=str, default='scoring_model/quality_scorer.pkl')
    parser.add_argument('--perplexity_model', type=str, default='distilgpt2')
    parser.add_argument('--toxicity_model', type=str, default='unitary/toxic-bert')
    parser.add_argument('--threshold', type=float, default=0.75, help="Minimum quality score to keep a document.")
    
    args = parser.parse_args()
    run_pipeline(args)
