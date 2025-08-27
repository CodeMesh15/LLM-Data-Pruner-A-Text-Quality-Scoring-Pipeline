
import torch
import numpy as np
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- Heuristic Metrics ---

def calculate_repetition_rate(text, n=3):
    """Calculates the percentage of duplicate n-grams."""
    if not text: return 0.0
    words = text.lower().split()
    if len(words) < n: return 0.0
    
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    if not ngrams: return 0.0
    
    ngram_counts = Counter(ngrams)
    duplicate_count = sum(c - 1 for c in ngram_counts.values())
    return duplicate_count / len(ngrams)

def calculate_symbol_ratio(text, symbols=['#', '{', '}', '==', '->']):
    """Calculates the ratio of specified symbols to words."""
    if not text: return 0.0
    words = text.split()
    if not words: return 0.0
    
    symbol_count = sum(text.count(s) for s in symbols)
    return symbol_count / len(words)

# --- Model-Based Metrics ---

def calculate_perplexity(text, model, tokenizer, device="cpu"):
    """Calculates the perplexity of a text using a causal language model."""
    if not text: return float('inf')
    
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        
    # The loss is the negative log likelihood, which is the cross-entropy loss.
    # Perplexity is the exponential of this loss.
    neg_log_likelihood = outputs.loss.item()
    return np.exp(neg_log_likelihood)

def calculate_toxicity(text, model, tokenizer, device="cpu"):
    """Calculates the toxicity score of a text."""
    if not text: return 0.0
    
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # The model outputs logits for 'toxic' and 'non-toxic'.
    # We apply softmax to get probabilities.
    probabilities = torch.softmax(logits, dim=1)
    # Return the probability of the 'toxic' class (usually index 1)
    return probabilities[0][1].item()

# --- Main Wrapper Function ---

def compute_all_metrics(text, models):
    """Computes all defined metrics for a given text."""
    metrics = {
        'repetition_rate': calculate_repetition_rate(text),
        'symbol_ratio': calculate_symbol_ratio(text),
        'perplexity': calculate_perplexity(
            text, models['perplexity_model'], models['perplexity_tokenizer'], models['device']
        ),
        'toxicity': calculate_toxicity(
            text, models['toxicity_model'], models['toxicity_tokenizer'], models['device']
        )
    }
    return metrics

if __name__ == '__main__':
    # --- Example Usage ---
    print("Loading models for demonstration...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load perplexity model (a smaller GPT-2 is good for this)
    perp_model_name = "distilgpt2"
    perplexity_model = AutoModelForCausalLM.from_pretrained(perp_model_name).to(device)
    perplexity_tokenizer = AutoTokenizer.from_pretrained(perp_model_name)
    
    # Load toxicity model
    tox_model_name = "unitary/toxic-bert"
    toxicity_model = AutoModelForSequenceClassification.from_pretrained(tox_model_name).to(device)
    toxicity_tokenizer = AutoTokenizer.from_pretrained(tox_model_name)
    
    models = {
        'perplexity_model': perplexity_model,
        'perplexity_tokenizer': perplexity_tokenizer,
        'toxicity_model': toxicity_model,
        'toxicity_tokenizer': toxicity_tokenizer,
        'device': device
    }
    
    # --- Test with sample texts ---
    high_quality_text = "The quick brown fox jumps over the lazy dog. This sentence is a classic pangram used in typography and design."
    low_quality_text = "error error error ==>> click here click here http://spam.com/lol {#$#$#}"
    
    print("\n--- High Quality Text ---")
    metrics_high = compute_all_metrics(high_quality_text, models)
    for key, value in metrics_high.items():
        print(f"{key}: {value:.4f}")
        
    print("\n--- Low Quality Text ---")
    metrics_low = compute_all_metrics(low_quality_text, models)
    for key, value in metrics_low.items():
        print(f"{key}: {value:.4f}")
