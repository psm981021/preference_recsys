
python main.py \
    --model_name UPTRec \
    --data_name Video_Games  \
    --data_dir data/10core \
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
    --output_dir Best/Video_Games/7\
    --model_idx Video \
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 0.1 \
    --num_intent_clusters 2\
    --intent_cf_weight 1.2 \
    --cf_weight 0 \
    --cluster_value 0.3 \
    --ncl \
    --num_hidden_layers 1 \


# ./scripts/main_table.sh