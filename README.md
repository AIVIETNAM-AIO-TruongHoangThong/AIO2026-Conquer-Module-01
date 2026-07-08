# Similarity Matching Methods for Images and Texts

This repository contains a project that compares different methods for matching images and texts. It tests three ways to solve the **Semantic Gap**, which is the problem that images and text naturally belong to different mathematical spaces.

## Project Overview

[![Project Video](https://img.youtube.com/vi/CljX-IhiE7w/maxresdefault.jpg)](https://youtu.be/CljX-IhiE7w)

The main question of this project is:
> *How much better is CLIP compared to traditional methods (like feature engineering and linear projection) for matching images with text?*

We test three methods on the **Flickr8k** dataset:
1. **Basic Baseline:** Comparing ResNet50 image features and TF-IDF text features directly.
2. **Linear Projection:** Using a simple machine learning model (`LinearRegression`) to learn how to map image features into text features.
3. **Contrastive Language-Image Pretraining (CLIP):** Using a pre-trained model that maps both images and text into a shared space.

## Key Features and Metrics

The project uses the following evaluation methods:
- **Fast Metrics:** Calculates Recall@1, Recall@5, and Mean Reciprocal Rank (MRR) quickly using NumPy matrix multiplication.
- **Three Types of Tests:** Evaluates the models on Positive Pairs, Random Negatives, and Hard Negatives (where key words are swapped, like changing "dog" to "cat").
- **Ablation Studies:** Tests how changing the TF-IDF vocabulary size (500, 1000, and 2000) affects the results.
- **Visual Analysis:** Creates heatmaps, shows the top text retrieval examples, and analyzes errors to see why the models fail.

## Setup and Usage

This project uses `uv` to manage packages and `jupytext` to sync between the `.py` script and the `.ipynb` notebook.

### 1. Install dependencies

This project requires **Python 3.13+**. Make sure you have `uv` installed, then set up the environment. This installs PyTorch, Transformers, Datasets, scikit-learn, and the plotting/table dependencies:
```bash
uv sync
```
> A CUDA-capable GPU is recommended. Feature extraction runs on the GPU automatically when one is available and falls back to CPU otherwise.

### 2. Run the Notebook
You can open `AIO2026_CONQUER_MODULE_1.ipynb` directly in Jupyter, Google Colab, or VS Code to run the code step-by-step.

Alternatively, if you want to run the Python script:
```bash
uv run python aio2026_conquer_module_1.py
```

### 3. Sync Changes (Optional)
If you edit the Python file and want to update the notebook, use `--update` to preserve the existing cell outputs:
```bash
uv run jupytext --update --to notebook aio2026_conquer_module_1.py -o AIO2026_CONQUER_MODULE_1.ipynb
```
> Plain `--to notebook` (without `--update`) rewrites the notebook and **discards all saved outputs**.

## Results Summary

Tested on 1,000 image-caption pairs from Flickr8k:

| Method | Recall@1 | Recall@5 | MRR | Hard Negative Score |
|---|---|---|---|---|
| Basic Baseline | 0.2% | 0.5% | 0.0075 | 0.0329 |
| Linear Projection | 5.5% | 17.0% | 0.1190 | 0.1635 (Failed) |
| CLIP | **55.0%** | **83.2%** | **0.6736** | 0.2876 (Passed, but struggles) |

**Conclusion:** Basic linear projection creates a false sense of alignment but fails completely on hard negatives. CLIP performs much better because it learns a strong, shared space for both images and text.

## License
Created as part of the AI VIET NAM - AI Conquer 2026 program.
