# python main.py \
#     --model_name UPTRec \
#     --data_name Beauty  \
#     --data_dir data/5core \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 0 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Ablation/Beauty/Item_level/29 \
#     --model_idx Mean \
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 1 \
#     --num_intent_clusters 2\
#     --intent_cf_weight 1.4\
#     --cf_weight 0 \
#     --cluster_value 0.3 \
#     --cluster_prediction \
#     --num_hidden_layers 1 \

# python main.py \
#     --model_name UPTRec \
#     --data_name Beauty  \
#     --data_dir data/5core \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 0 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Ablation/Beauty/Item_level/30 \
#     --model_idx Mean \
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1.2 \
#     --temperature 1 \
#     --num_intent_clusters 2\
#     --intent_cf_weight 1.4\
#     --cf_weight 0 \
#     --cluster_value 0.3 \
#     --cluster_prediction \

# python main.py \
#     --model_name UPTRec \
#     --data_name Beauty  \
#     --data_dir data/5core \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 0 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Ablation/Beauty/Item_level/31 \
#     --model_idx Mean \
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 1 \
#     --num_intent_clusters 2\
#     --intent_cf_weight 1.4\
#     --cf_weight 0.2 \
#     --cluster_value 0.3 \
#     --cluster_prediction \

# python main.py \
#     --model_name UPTRec \
#     --data_name Beauty  \
#     --data_dir data/5core \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 0 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Ablation/Beauty/Item_level/32 \
#     --model_idx Mean \
#     --contrast_type Item-Level \
#     --augment_type crop \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 1 \
#     --num_intent_clusters 2\
#     --intent_cf_weight 1.4\
#     --cf_weight 0 \
#     --cluster_value 0.3 \
#     --cluster_prediction \

python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --data_dir data/5core \
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
    --output_dir Ablation/Beauty/Item_level/33 \
    --model_idx Mean \
    --contrast_type Item-Level \
    --augment_type crop \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 2\
    --intent_cf_weight 1.4\
    --cf_weight 0 \
    --cluster_value 0.3 \
    --cluster_prediction \
    --cluster_temperature \
    --ncl \
    --bi_direction \
    --position_encoding_false

python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --data_dir data/5core \
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
    --output_dir Ablation/Beauty/Item_level/34 \
    --model_idx Mean \
    --contrast_type Item-Level \
    --augment_type crop \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 2\
    --intent_cf_weight 1.4\
    --cf_weight 0 \
    --cluster_value 0.3 \
    --cluster_prediction \
    --cluster_temperature \
    --ncl \

# scripts/Beauty/0.sh