
python main.py \
    --model_name UPTRec \
    --data_name Toys_and_Games  \
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
    --output_dir Main_Table/Toys/Item_level/5 \
    --model_idx Mean \
    --num_intent_clusters 5 \
    --cluster_value 0.3 \
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --intent_cf_weight 0.1\
    --cf_weight 0

python main.py \
    --model_name UPTRec \
    --data_name Toys_and_Games  \
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
    --output_dir Main_Table/Toys/Item_level/10 \
    --model_idx Mean \
    --num_intent_clusters 10 \
    --cluster_value 0.3 \
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --intent_cf_weight 0.1\
    --cf_weight 0

python main.py \
    --model_name UPTRec \
    --data_name Toys_and_Games  \
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
    --output_dir Main_Table/Toys/Item_level/25 \
    --model_idx Mean \
    --num_intent_clusters 5 \
    --cluster_value 0.3 \
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --intent_cf_weight 0.1\
    --cf_weight 0

# scripts/toys.sh