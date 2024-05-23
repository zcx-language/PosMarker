#!/usr/bin/env bash
conda activate pytorch1.0
cd /data_ssd2/hzh/PytorchSSD1215
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=1 python test_demo.py -e 10
CUDA_VISIBLE_DEVICES=1 python test_demo.py -e 15
CUDA_VISIBLE_DEVICES=1 python test_demo.py -e 20
CUDA_VISIBLE_DEVICES=1 python test_demo.py -e 25
CUDA_VISIBLE_DEVICES=1 python test_demo.py -e 30
CUDA_VISIBLE_DEVICES=1 python test_demo.py -e 35
CUDA_VISIBLE_DEVICES=1 python test_demo.py -e 40
CUDA_VISIBLE_DEVICES=1 python test_demo.py -e 45
CUDA_VISIBLE_DEVICES=1 python test_demo.py -e 50
CUDA_VISIBLE_DEVICES=1 python test_demo.py -e 55
CUDA_VISIBLE_DEVICES=1 python test_demo.py -e 60


