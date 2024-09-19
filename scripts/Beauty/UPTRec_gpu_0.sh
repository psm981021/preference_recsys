
python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --context encoder \
    --seq_representation_type concatenate \
    --attention_type Cluster \
    --cluster_joint \
    --de_noise \
    --batch_size 512 \
    --epochs 2000 \
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 30 \
    --embedding \
    --output_dir output/Beauty/Contrastive_Learning/Item_level/test \
    --model_idx test\
    --contrast_type Item-level \
    --cluster_train 10 \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 0.1 \
    --num_intent_clusters 10\
    --intent_cf_weight 2\
    --cf_weight 1 \
    --cluster_value 0.7 \

# scripts/Beauty/UPTRec_gpu_0.sh