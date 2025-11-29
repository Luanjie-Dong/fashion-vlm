# Fashion Recommender

## Overview
Advanced fashion recommendation system that combines Vision-Language Models (VLM) with graph-based embeddings to provide personalized outfit recommendations. The system understands clothing attributes from images and suggests compatible items based on both visual semantics and stylistic compatibility.

## Features
| Feature | Description | Status |
|---------|-------------|--------|
| **VLM Attribute Extraction** | Extract detailed clothing attributes and generate multimodal embeddings from images |  In Progress |
| **Graph-Based Compatibility** | Fashion item graph with learned compatibility relationships using Siamese networks | In Progress|
| **Personalized Recommendations** | Hybrid recommendations combining user profiles with vector and graph search | In Progress |
| **Multimodal Embedding Search** | Combined text and image embeddings for semantic fashion search | In Progress |

## Technical Implementations
- **VLM Processing Pipeline** - [View Code](src/vlm_processor.py)
  - Utilizes BLIP-2/FashionCLIP for attribute extraction from images
  - VLM model is finetuned on Deep Fashion multimodal dataset
  - Generates both text and image embeddings for comprehensive item representation
  - Implements multimodal embedding fusion for robust semantic understanding

- **Compatibility Graph Engine** - [View Code](src/graph_engine.py)
  - Constructs fashion item nodes with category and attribute metadata
  - Implements Siamese networks trained on Polyvore outfit dataset
  - Supports both rule-based and learned compatibility relationships

- **Hybrid Recommendation System** - [View Code](src/recommender.py)
  - Combines vector similarity search with graph traversal
  - Integrates user profile vectors for personalized results
  - Implements diversity-aware ranking for varied recommendations

## References
- [Polyvore Dataset](https://github.com/xthan/polyvore-dataset) - Outfit compatibility dataset
- [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) - Multimodal Clothing attribute dataset

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

