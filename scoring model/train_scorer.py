
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# Import the feature calculation function
from feature_engineering.text_metrics import compute_all_metrics

def train_scorer(data_path='scoring_model/quality_training_set.csv', model_dir='scoring_model'):
    """
    Trains a LightGBM model to predict text quality.
    """
    print("--- Training the Quality Scoring Model ---")

    # --- 1. Load Models for Feature Calculation ---
    print("Loading feature calculation models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    perp_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    perp_tok = AutoTokenizer.from_pretrained("distilgpt2")
    tox_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert").to(device)
    tox_tok = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    
    models = {
        'perplexity_model': perp_model, 'perplexity_tokenizer': perp_tok,
        'toxicity_model': tox_model, 'toxicity_tokenizer': tox_tok,
        'device': device
    }
    
    # --- 2. Load and Featurize Labeled Data ---
    if not os.path.exists(data_path):
        print(f"Error: Training data not found at '{data_path}'. Please run create_training_set.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    print("Calculating features for the training set...")
    tqdm.pandas(desc="Featurizing")
    # Apply the compute_all_metrics function to each row
    feature_dicts = df['text'].progress_apply(lambda x: compute_all_metrics(str(x), models))
    
    features_df = pd.DataFrame(list(feature_dicts))
    
    # --- 3. Train the Classifier ---
    print("Training LightGBM classifier...")
    
    X = features_df
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    lgbm = lgb.LGBMClassifier(objective='binary', random_state=42)
    lgbm.fit(X_train, y_train)
    
    # --- 4. Evaluate and Save ---
    print("\n--- Model Evaluation ---")
    predictions = lgbm.predict(X_test)
    print(classification_report(y_test, predictions, target_names=['low_quality', 'high_quality']))
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'quality_scorer.pkl')
    joblib.dump(lgbm, model_path)
    
    print(f"--- Quality scoring model saved to '{model_path}' ---")

if __name__ == '__main__':
    train_scorer()
