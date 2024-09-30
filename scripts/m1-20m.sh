
python main.py \
    --model_name UPTRec \
    --data_name m1-20m \
    --data_dir data\
    --context encoder \
    --seq_representation_type mean \
    --attention_type Cluster \
    --lr 0.0001 \
    --cluster_joint \
    --de_noise \
    --batch_size 6000 \
    --epochs 2000 \
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 30 \
    --embedding \
    --output_dir Main_Table/m1-20m/Item-level/1\
    --model_idx V1 \
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 10\
    --intent_cf_weight  2\
    --cf_weight 0 \
    --cluster_value 0.3 \
    --use_multi_gpu \

# ./scripts/m1-20m.sh