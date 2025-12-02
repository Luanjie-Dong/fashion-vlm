# 👗 VLM Fashion Attribute Extractor

## Overview
Using a fine-tuned Vision-Language Model (VLM) to automatically extract detailed garment and outfit attributes directly from fashion images. 

## Features
| Feature | Description | Status |
|---------|-------------|--------|
| **VLM Attribute Extraction** | Extracts a structured, granular list of attributes (e.g., dress: a-line, floral, midi) and their properties from input images using a fine-tuned $\text{PaliGemma}$ model. | **✅ Completed** |
| **Efficient Fine-Tuning** | Uses LoRA (Low-Rank Adaptation) for faster, resource-efficient fine-tuning on the $\text{FashionPedia}$ dataset. | **✅ Completed** |
| **Semantic Fashion Search** | Leverages generated embeddings for semantic similarity search, allowing users to find visually and textually similar items based on extracted attributes. | **🚧 In Progress** |


## Technical Implementations
- **VLM Processing Pipeline** - [View Code]()
  - Utilizes PaliGemma for attribute extraction from images
  - VLM model is finetuned on FashionPedia multimodal dataset
    - LoRA used for faster finetuning
    - flash attention mode used for faster forward and backward pass




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