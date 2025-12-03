# 👗 VLM Fashion Attribute Extractor

## Overview
This project leverages a fine-tuned Vision-Language Model (VLM) to automatically extract detailed, structured attributes of garments and outfits directly from fashion images. It provides rich visual understanding for downstream applications such as semantic search, recommendation, or catalog enrichment.


## Features
| Feature | Description | Status |
|---------|-------------|--------|
| **VLM Attribute Extraction** | Extracts a structured, granular list of fashion attributes (e.g., `dress: a-line, floral, midi`) from input images using a fine-tuned **PaliGemma** model. | ✅ Completed      |
| **Gradio Frontend** | Provides a user‑friendly interface where users can upload any fashion image and instantly view the extracted attributes. | ✅ Completed |
| **Semantic Fashion Search** | Utilizes the generated embeddings to power a similarity search that retrieves visually and textually similar items based on the extracted attributes. | 🚧 In Progress |



## Technical Implementations
- **VLM Dataset prepration** - [View Code](https://github.com/Luanjie-Dong/fashion-vlm/blob/main/src/training/dataloader.py)
    - Reformatted a fashion dataset (Fashionpedia) for attribute extraction fine-tuning. - [code](https://github.com/Luanjie-Dong/fashion-recommender/blob/main/src/notebooks/preprocess_data.ipynb) 
    - Added a standardized prompt: 
     ```text
    <image> Describe all garments and their attributes in this image.
    ```
    - Built a custom `Dataset` and `DataLoader` that efficiently batch and encode multimodal inputs (images + text prompts) together with their labels.
- **VLM Fine‑Tuning** – [View Code](https://github.com/Luanjie-Dong/fashion-recommender/blob/main/src/training/train.py)
  - Fine‑tuned the `google/paligemma2-3b-pt-224` model on 34 k fashion images for attribute extraction.  
  - Employed LoRA (Low‑Rank Adaptation) for parameter‑efficient training on a single GPU.  
  - Leveraged a FlashAttention‑enabled version of PaliGemma to accelerate both training and inference.


## Running the Application
Follow these steps to launch the project locally:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Luanjie-Dong/fashion-vlm.git
   cd fashion-vlm
   ```

2. **Create a virtual environment and install dependencies**  
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Launch the Gradio frontend**  
   ```bash
   python display.py
   ```
   Open the provided URL to upload a photo and view the extracted attributes.

## Future Development
- Complete the **Semantic Fashion Search** pipeline, integrating the generated embeddings with a retrieval engine.
- Optimize inference latency on edge devices via quantization or efficient model variants.

## Contributors

<a href="https://github.com/Luanjie-Dong/fashion-vlm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Luanjie-Dong/fashion-vlm" />
</a>

# References 
- hugging face paligemma finetuning example - https://github.com/huggingface/notebooks/tree/main/examples/paligemma
- youtube paligemma finetuning for json example - https://www.youtube.com/watch?v=hDa-M91MSGU
- optimising training on single gpu - https://huggingface.co/docs/transformers/v4.42.0/perf_train_gpu_one
- FashionPedia dataset link - https://www.kaggle.com/competitions/imaterialist-fashion-2020-fgvc7/overview
