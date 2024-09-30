
python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --data_dir data \
    --context encoder \
    --seq_representation_type mean \
    --attention_type Cluster \
    --cluster_joint \
    --de_noise \
    --batch_size 512 \
    --epochs 2000 \
    --gpu_id 1 \
    --visualization_epoch 10 \
    --patience 30 \
    --embedding \
    --output_dir output/Beauty/Item_level/tests\
    --model_idx asdr \
    --contrast_type Item-Level \
    --augment_type mask \
    --n_views 3 \
    --cluster_train 1 \
    --warm_up_epoches 0\
    --num_intent_clusters 10 \
    --intent_cf_weight 0.01 \
    --cf_weight 1 \
    --num_hidden_layers 2 \
    --gamma 0.3 \
    --num_user_intent_clusters 256 \
    --intent_cf_user_weight 0.9 \
    --temperature 1 \
    --de_noise \
    --use_multi_gpu

# scripts/Beauty/Cluster_Attention_Hybrid_gpu_0.sh