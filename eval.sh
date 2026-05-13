#!/bin/bash
# Vison + Text evaluation
python evaluate_qwen.py \
    --csv        outputs/mocheg_synthetic_12k.csv \
    --images_dir mocheg/train/images \
    --mode       vision \
    --model      Qwen/Qwen2.5-VL-7B-Instruct \
    --no_nei
    --output     outputs/results_vision.csv

#Text evaluation
python evaluate_qwen.py \
    --csv        outputs/mocheg_synthetic_12k.csv \
    --mode       text \
    --model      Qwen/Qwen2.5-7B-Instruct \
    --no_nei
    --output     outputs/results_text.csv

# Vison + Text evaluation with No Evidence
python evaluate_qwen.py \
    --csv        outputs/mocheg_synthetic_12k.csv \
    --images_dir mocheg/train/images \
    --mode       vision \ 
    --model      Qwen/Qwen2.5-VL-7B-Instruct \
    --no_nei \
    --no_evidence \
    --output     outputs/results_vision_no_evidence.csv

#Text evaluation with No Evidence
python evaluate_qwen.py \
    --csv        outputs/mocheg_synthetic_12k.csv \
    --mode       text \
    --model      Qwen/Qwen2.5-7B-Instruct \
    --no_nei \
    --no_evidence \
    --output     outputs/results_text_no_evidence.csv