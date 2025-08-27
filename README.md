# LLM-Data-Pruner: A Text Quality Scoring Pipeline

An implementation of a data pruning pipeline that scores text documents based on quality and aesthetic metrics. This system is designed to filter large datasets to improve the efficiency and performance of Large Language Model (LLM) training, inspired by data curation work at Meta AI.

---

## 1. Project Overview

The performance of Large Language Models is heavily dependent on the quality of the training data. This project replicates the creation of a "text quality aesthetic score pipeline" to perform effective data pruning. The goal is to filter massive, raw text corpora to retain only high-quality documents. As noted in the source experience, such a pipeline can allow models to achieve state-of-the-art performance using significantly less data.

---

## 2. Core Objectives

-   To define a set of heuristics and model-based metrics to quantify text quality.
-   To build a scoring model that outputs a single "aesthetic score" for a given document.
-   To create an efficient, scalable pipeline that can process and filter a large text dataset.
-   To demonstrate the effectiveness of the pruned dataset by training a small language model.

---

## 3. Methodology

#### Phase 1: Defining Quality Metrics

We'll create a feature engineering script that calculates a variety of metrics for any given text document. These will include:

-   **Heuristic-Based Metrics**:
    -   Text length and average word length.
    -   Symbol-to-word ratio (e.g., high ratio of `#` or `{}` might indicate code or messy text).
    -   Repetition rate (percentage of duplicate n-grams).
    -   Stopword percentage.
-   **Model-Based Metrics**:
    -   **Perplexity score** from a pre-trained masked language model (e.g., DistilBERT). High-quality, natural language should have low perplexity.
    -   **Toxicity score** using a pre-trained toxicity detection model.

#### Phase 2: Building a Scoring Model

While heuristics are useful, a single "aesthetic score" is more powerful.

1.  **Dataset Creation**: We'll create a labeled dataset by taking text from known high-quality sources (e.g., Wikipedia, academic papers) and low-quality sources (e.g., noisy web scrapes, forum comments).
2.  **Model Training**: We'll train a simple classifier (like LightGBM or Logistic Regression) on the features from Phase 1 to predict a binary label: `high_quality` (1) or `low_quality` (0). The output probability of this model will serve as our "aesthetic score."

#### Phase 3: The Pruning Pipeline

This will be a script that can process a large, raw text file (e.g., a shard from the [C4 dataset](https://www.tensorflow.org/datasets/catalog/c4)).

1.  The script will read documents one by one.
2.  For each document, it will calculate all the heuristic metrics and the model-based aesthetic score.
3.  It will apply a set of filtering rules (e.g., `aesthetic_score > 0.8` AND `repetition_rate < 0.1`).
4.  Documents that pass the filter will be written to a new, clean output file.

#### Phase 4: Demonstration of Impact

To prove the pipeline works, we'll show its effect on a small language model.

1.  Take a small slice of a public dataset (e.g., 100MB of C4).
2.  Use our pipeline to create a pruned version (which might be ~50MB, mirroring the 50% reduction mentioned in the resume).
3.  Train a small GPT-style model (e.g., using a framework like nanoGPT) on **both** the original and the pruned data for the same number of steps.
4.  Compare the validation loss of the two models. The model trained on the pruned data should achieve a lower loss, demonstrating that we've improved data quality.

