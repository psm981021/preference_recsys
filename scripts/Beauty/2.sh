
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
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 30 \
    --embedding \
    --output_dir Ablation/Beauty/Item_level/Max_seq/60 \
    --model_idx Mean\
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 5\
    --intent_cf_weight 1\
    --cf_weight 0 \
    --cluster_value 0.3\
    --max_seq_length 60\

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
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 30 \
    --embedding \
    --output_dir Ablation/Beauty/Item_level/Max_seq/70 \
    --model_idx Mean\
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 5\
    --intent_cf_weight 1\
    --cf_weight 0 \
    --cluster_value 0.3\
    --max_seq_length 70\

# scripts/Beauty/2.sh