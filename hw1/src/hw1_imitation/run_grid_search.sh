#!/bin/bash
# Chạy từ thư mục gốc của project
for lr in 1e-3 3e-4 1e-4; do
    echo "--- Dang chay voi Learning Rate: $lr ---"
    # Dùng cd .. để nhảy ra ngoài thư mục gốc rồi mới chạy python
    cd ../.. && uv run python src/hw1_imitation/train.py \
        --policy_type flow \
        --lr $lr \
        --eval_interval 50000 \
        --exp_name "lr_sweep_$lr" && cd src/hw1_imitation
done
