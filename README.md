# Similarity Matching Methods for Images and Texts

This repository contains an empirical study exploring different methods for cross-modal similarity matching between images and texts. The project evaluates three generations of approaches to solve the **Semantic Gap**—the fundamental disconnect between visual and textual mathematical spaces.

## 🚀 Project Overview

The core hypothesis tested in this project is:
> *How much better are modern multimodal embedding models (CLIP) than traditional feature engineering and linear projection for image-text similarity matching?*

We evaluate three distinct methods on the **Flickr8k** dataset:
1. **Naive Baseline:** Directly comparing independently extracted ResNet50 visual features and TF-IDF text features.
2. **Linear Projection:** Using a shallow machine learning model (`LinearRegression`) to learn a projection matrix from the visual space to the textual space.
3. **Contrastive Language-Image Pretraining (CLIP):** Using a foundation model that naturally maps both modalities into a shared embedding space.

## 📊 Key Features & Metrics

The experimental pipeline is designed as a strict evaluation framework:
- **Fast, Vectorized Metrics:** Calculates optimized Recall@1, Recall@5, and Mean Reciprocal Rank (MRR) using NumPy matrix multiplication.
- **Three-Tiered Evaluation:** Tests matching quality against Positive Pairs, Random Negatives, and synthetically generated **Hard Negatives** (swapping key subjects like "dog" $\rightarrow$ "cat").
- **Ablation Studies:** Explores the impact of varying the TF-IDF vocabulary size (500 vs. 1000 vs. 2000).
- **Qualitative Analysis:** Generates visual heatmaps, top-K text retrieval examples for query images, and automated Error Analysis to understand model failure modes.

## 🛠 Setup and Usage

This project uses `uv` for lightning-fast dependency management and `jupytext` for syncing between `.py` and `.ipynb` files.

### 1. Install dependencies

Ensure you have `uv` installed, then set up the environment:
```bash
uv sync
```

### 2. Run the Notebook
You can open `AIO2026_CONQUER_MODULE_1.ipynb` directly in Jupyter, Google Colab, or VS Code to run the full pipeline step-by-step.

Alternatively, if you want to run the raw Python script:
```bash
uv run python aio2026_conquer_module_1.py
```

### 3. Sync Changes (Optional)
If you edit the Python file and want to update the notebook:
```bash
uv run jupytext --to notebook aio2026_conquer_module_1.py -o AIO2026_CONQUER_MODULE_1.ipynb
```

## 📈 Results Summary

Tested on 1,000 image-caption pairs from Flickr8k:

| Method | Recall@1 | Recall@5 | MRR | Hard Negative Score |
|---|---|---|---|---|
| Naive Baseline | 0.2% | 0.5% | 0.0075 | 0.0329 |
| Linear Projection | 5.5% | 17.0% | 0.1190 | 0.1635 (Failed) |
| CLIP | **55.0%** | **83.2%** | **0.6736** | 0.2876 (Passed, but struggles) |

**Conclusion:** Shallow linear mappings create an illusion of alignment but fail catastrophically on hard negatives. True multimodal contrastive learning (CLIP) vastly outperforms traditional methods by learning a robust, shared semantic space.

## 📝 License
Created as part of the AI VIET NAM - AI Conquer 2026 program.
