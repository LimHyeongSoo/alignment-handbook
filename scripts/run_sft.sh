#!/bin/bash

## 1) CUDA device set
export CUDA_VISIBLE_DEVICES="1"

## Hugging Face 캐시 위치 설정
export HF_HOME=~/PycharmProjects/alignment-handbook/cache/

## Accelerate 로그 레벨 설정 (선택 사항)
export ACCELERATE_LOG_LEVEL=info

## Over 2GPU
#ACCELERATE_USE_FSDP=true accelerate launch \
#--config_file ~/PycharmProjects/alignment-handbook/recipes/accelerate_configs/fsdp.yaml \
#--num_processes 1 \
#~/PycharmProjects/alignment-handbook/scripts/run_sft.py \
#~/PycharmProjects/alignment-handbook/recipes/smollm2/sft/config.yaml \
#--per_device_train_batch_size=1 \
#--num_train_epochs=2

# Single GPU
accelerate launch \
--config_file ~/PycharmProjects/alignment-handbook/recipes/accelerate_configs/single_gpu.yaml \
--num_processes=1 \
~/PycharmProjects/alignment-handbook/scripts/run_sft.py \
~/PycharmProjects/alignment-handbook/recipes/AMD-OLMO/sft/config.yaml \
--per_device_train_batch_size=8 \
--num_train_epochs=3
