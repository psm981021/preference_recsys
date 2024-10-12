python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --data_dir data/ \
    --context encoder \
    --seq_representation_type mean \
    --attention_type Cluster \
    --cluster_joint \
    --de_noise \
    --batch_size 256 \
    --epochs 2000 \
    --gpu_id 0 \
    --visualization_epoch 20 \
    --patience 30 \
    --embedding \
    --output_dir Ablation/Beauty/Item_level/24 \
    --model_idx Mean \
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 2\
    --intent_cf_weight 1.2\
    --cf_weight 1 \
    --cluster_value 0.3 \
    --cluster_prediction

# ./scripts/Beauty/test_1.sh 