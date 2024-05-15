
python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --output_dir renew_log/Beauty/Cluster-Attention-16-1-concatenate/\
    --contrast_type Hybrid  \
    --context encoder \
    --seq_representation_type concatenate \
    --attention_type Cluster \
    --batch_size 512 \
    --epochs 2000 \
    --patience 200 \
    --warm_up_epoches 0 \
    --num_intent_clusters 16 \
    --intent_cf_weight 0.1 \
    --cf_weight 0.1 \
    --num_hidden_layers 1 \
    --model_idx Cluster-Attention-0.1 \
    --gpu_id 0 \
    --embedding \
    --visualization_epoch 10 \
    --temperature=1 \
    --cluster_valid \ 
    --wandb

# scripts/Beauty/Cluster_Attention_Hybrid_gpu_0.sh
# --temperature=0.1 \
# --num_hidden_layers 2 \
# --cluster_train 1 