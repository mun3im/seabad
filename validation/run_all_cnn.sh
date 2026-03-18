#!/bin/bash

START_TIME=$(date +%s)

DATASET_ROOT="/Volumes/Evo/SEABAD"

if [ $# -gt 0 ]; then
    SEEDS=("$@")
else
    SEEDS=(42 100 786)
fi

for seed in "${SEEDS[@]}"; do
    for model in mobilenetv3s efficientnetb0 resnet50 vgg16; do
        python validate_seabad_pretrained.py \
            --model $model \
            --seed $seed
        echo "  Done."
    done
done


END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "All runs complete!"
echo "Total wall time: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s"
