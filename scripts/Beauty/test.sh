
python3 main.py \
    --data_name Beauty --cf_weight 0.1 \
    --model_idx UPTRec-test --gpu_id 1 \
    --batch_size 512 --contrast_type Hybrid \
    --output_dir renew_log/Beauty/UPTRec-test \
    --context item_embedding \
    --attention_type Base \
    --num_intent_cluster 8 --seq_representation_type mean \
    --epochs=2000 \
    --warm_up_epoches 0 --intent_cf_weight 0.1 --num_hidden_layers 1 --temperature=0.5 \

# scripts/Beauty/test.sh
# --temperature=0.1 \