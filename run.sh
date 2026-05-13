#!/bin/bash

python generate_dataset_clip.py \
    --corpus2     mocheg/train/Corpus2.csv \
    --img_qrels   mocheg/train/img_evidence_qrels.csv \
    --images_dir  mocheg/train/images \
    --output      outputs/mocheg_synthetic_12k.csv \
    --target      3000 \
    --spacy_model en_core_web_trf \
    --clip_model  ViT-L/14 \
    --top_k       10 \
    --embed_cache outputs/clip_cache.pkl