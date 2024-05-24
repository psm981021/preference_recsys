python3 main.py \
    --data_name Beauty --cf_weight 0.1 \
    --output_dir output_working/3 \
    --model_idx 3 --gpu_id 1 \
    --batch_size 512 --contrast_type Hybrid \
    --num_intent_cluster 8 --seq_representation_type mean \
    --n_views 3 \
    --warm_up_epoches 0 --intent_cf_weight 0.1 --num_hidden_layers 1 \

# scripts/run_beauty.sh