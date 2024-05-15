
python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --output_dir renew_log/Beauty/Test/visual/ \
    --contrast_type Hybrid  \
    --context encoder \
    --seq_representation_type mean \
    --attention_type Cluster \
    --batch_size 512 \
    --epochs 1000 \
    --patience 50 \
    --warm_up_epoches 0 \
    --num_intent_clusters 16 \
    --intent_cf_weight 0.1 \
    --cf_weight 0.1 \
    --num_hidden_layers 1 \
    --model_idx Test-Self-Attention-test_visual \
    --gpu_id 1 \
    --embedding \

# scripts/Beauty/test.sh
# --temperature=0.1 \