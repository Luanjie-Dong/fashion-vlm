# 👗 VLM Fashion Attribute Extractor

## Overview
This project leverages a fine-tuned Vision-Language Model (VLM) to automatically extract detailed, structured attributes of garments and outfits directly from fashion images. It provides rich visual understanding for downstream applications such as semantic search, recommendation, or catalog enrichment.


## Features
| Feature | Description | Status |
|---------|-------------|--------|
| **VLM Attribute Extraction** | Extracts a structured, granular list of fashion attributes (e.g., `dress: a-line, floral, midi`) from input images using a fine-tuned **PaliGemma** model. | ✅ Completed      |
| **Semantic Fashion Search**  | Uses generated embeddings to power semantic similarity search—enabling retrieval of visually and textually similar items based on extracted attributes. | 🚧 In Progress    |


## Technical Implementations
- **VLM Dataset prepration** - [View Code](https://github.com/Luanjie-Dong/fashion-recommender/blob/main/src/training/dataloader.pyb)
    - Reformatted a fashion dataset (Fashionpedia) for attribute extraction fine-tuning. - [code](https://github.com/Luanjie-Dong/fashion-recommender/blob/main/src/notebooks/preprocess_data.ipynb) 
    - Added a standardized prompt: 
     ```text
    <image> Describe all garments and their attributes in this image.
    ```
    - Implemented a custom Dataset and DataLoader to efficiently batch and encode multimodal inputs (images + text prompts) and corresponding labels.
- **VLM Finetuning** - [View Code](https://github.com/Luanjie-Dong/fashion-recommender/blob/main/src/training/train.py)
    - Fine-tuned google/paligemma2-3b-pt-224 for fashion attribute extraction.
    - Applied LoRA (Low-Rank Adaptation) for parameter-efficient training on a single GPU.
    - Utilized a FlashAttention-enabled version of PaliGemma to accelerate training and reduce memory overhead.




<!-- ## Implementation Phases
- Phase 1: Build Foundation
    - Implement Feature 1 with BLIP-2 or FashionCLIP
    - Create vector database with item embeddings
    - Basic similar-item recommendations
    - Dataset
        - Deepfashion 2

- Phase 2: Add Intelligence
    - Build graph structure with category compatibility rules
    - Implement user profile from liked items
    - Combine vector + graph search
    - Dataset
        - polyvore dataset

- Phase 3: Advanced Features
    - Train compatibility model on outfit data
    - Add style-based personalization
    - Optimize ranking with diversity -->

# References 
- hugging face paligemma finetuning example - https://github.com/huggingface/notebooks/tree/main/examples/paligemma
- youtube paligemma finetuning for json example - https://www.youtube.com/watch?v=hDa-M91MSGU
- optimising training on single gpu - https://huggingface.co/docs/transformers/v4.42.0/perf_train_gpu_one