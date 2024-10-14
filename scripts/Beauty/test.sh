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
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 30 \
    --embedding \
    --output_dir Ablation/Beauty/Item_level/35 \
    --model_idx Mean \
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 10\
    --intent_cf_weight 1.2\
    --cf_weight 0 \
    --cluster_value 0.3 \
    --cluster_prediction \
    --cluster_tempearture \
    --ncl \


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
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 30 \
    --embedding \
    --output_dir Ablation/Beauty/Item_level/36 \
    --model_idx Mean \
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 0.1 \
    --num_intent_clusters 10\
    --intent_cf_weight 1.2\
    --cf_weight 0 \
    --cluster_value 0.3 \
    --cluster_prediction \
    --ncl \

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
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 30 \
    --embedding \
    --output_dir Ablation/Beauty/Item_level/37 \
    --model_idx Mean \
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 0.1 \
    --num_intent_clusters 10\
    --intent_cf_weight 1.2\
    --cf_weight 0 \
    --cluster_value 0.3 \
    --cluster_tempearture \
    --ncl \



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
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Ablation/Beauty/Item_level/36 \
#     --model_idx Mean \
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 0.1 \
#     --num_intent_clusters 10\
#     --intent_cf_weight 1.2\
#     --cf_weight 0 \
#     --cluster_value 0.3 \
#     --cluster_prediction \
#     --ncl \
#     --lr 0.0001 \

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
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Ablation/Beauty/Item_level/38 \
#     --model_idx Mean \
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 0.1 \
#     --num_intent_clusters 10\
#     --intent_cf_weight 1.2\
#     --cf_weight 0 \
#     --cluster_value 0.3 \
#     --cluster_prediction \
#     --ncl \
#     --lr 0.0001 \
#     --bi_direction

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
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Ablation/Beauty/Item_level/37 \
#     --model_idx Mean \
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 0.1 \
#     --num_intent_clusters 10\
#     --intent_cf_weight 1.2\
#     --cf_weight 0 \
#     --cluster_value 0.3 \
#     --cluster_prediction \
#     --ncl \
#     --lr 0.0001 \
#     --bi_direction \
#     --augment_type crop \


# ./scripts/Beauty/test.sh 