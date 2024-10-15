

python main.py \
    --model_name UPTRec \
    --data_name ml-1m  \
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
    --output_dir Main_Table/ML-1M/Item_level/1 \
    --model_idx Mean\
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 10\
    --intent_cf_weight 1\
    --cf_weight 0 \
    --cluster_value 0.3\
    --max_seq_length 200\
    --cluster_prediction \

python main.py \
    --model_name UPTRec \
    --data_name ml-1m  \
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
    --output_dir Main_Table/ML-1M/Item_level/2 \
    --model_idx Mean\
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 10\
    --intent_cf_weight 1\
    --cf_weight 0 \
    --cluster_value 0.7\
    --max_seq_length 200\
    --cluster_prediction \

python main.py \
    --model_name UPTRec \
    --data_name ml-1m  \
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
    --output_dir Main_Table/ML-1M/Item_level/3 \
    --model_idx Mean\
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 5\
    --intent_cf_weight 1\
    --cf_weight 0 \
    --cluster_value 0.7\
    --max_seq_length 200\
    --cluster_prediction \

python main.py \
    --model_name UPTRec \
    --data_name ml-1m  \
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
    --output_dir Main_Table/ML-1M/Item_level/4 \
    --model_idx Mean\
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 5\
    --intent_cf_weight 1\
    --cf_weight 0 \
    --cluster_value 0.3\
    --max_seq_length 200\
    --cluster_prediction \

# ./scripts/ml-1m.sh 