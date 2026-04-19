#!/bin/bash
set -e

export HF_HOME=/Users/anshmadan/Desktop/headline-classifier/.hf_cache

echo "========================================"
echo "Starting Normal Classifier Training"
echo "========================================"
cd /Users/anshmadan/Desktop/headline-classifier
.venv/bin/python scripts/train_enhanced.py --data hierarchical-classifier/Dataset/BERT_training_V3.json --output-dir results_v3_normal > normal_train_v3.log 2>&1
echo "Finished Normal Classifier Training. Logs saved to normal_train_v3.log"

echo ""
echo "========================================"
echo "Starting Hierarchical Classifier Training"
echo "========================================"
cd /Users/anshmadan/Desktop/headline-classifier/hierarchical-classifier
../.venv/bin/python scripts/train_all.py --data Dataset/BERT_training_V3.json --output-dir results/v3_hierarchical > hier_train_v3.log 2>&1
echo "Finished Hierarchical Classifier Training. Logs saved to hier_train_v3.log"
echo "ALL DONE!"
