python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
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
    --output_dir output/Beauty/Contrastive_Learning/Item-User/test\
    --model_idx t12\
    --contrast_type Item-User \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 0.1 \
    --num_intent_clusters 10\
    --intent_cf_weight  0.1\
    --intent_cf_user_weight 0.9 \
    --cf_weight 0 \
    --cluster_value 0.3 \

# scripts/Beauty/test.sh     --cluster_temperature