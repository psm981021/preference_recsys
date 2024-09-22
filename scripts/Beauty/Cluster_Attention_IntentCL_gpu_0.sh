

## Amazon Sports

# python main.py \
#     --model_name UPTRec \
#     --data_name Sports_and_Outdoors  \
#     --data_dir data/ \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 40 \
#     --embedding \
#     --output_dir Main_Table/Amazon_Sports/Item-Level  \
#     --model_idx V-Item-Level\
#     --contrast_type Item-Level \
#     --cluster_train 10 \
#     --warm_up_epoches 0\
#     --num_intent_clusters 10 \
#     --num_hidden_layers 2 \
#     --cluster_value 0.3 \
#     --intent_cf_weight 10\
#     --cf_weight 0 \


## ML-1M

python main.py \
    --model_name UPTRec \
    --data_name ml-1m  \
    --data_dir data \
    --context encoder \
    --seq_representation_type mean \
    --attention_type Cluster \
    --cluster_joint \
    --de_noise \
    --batch_size 256 \
    --epochs 2000 \
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 40 \
    --embedding \
    --max_seq_length 200 \
    --output_dir Main_Table/ml-1m/Item-Level \
    --model_idx V-Item-Level\
    --contrast_type Item-Level \
    --cluster_train 10 \
    --num_hidden_layers 2 \
    --warm_up_epoches 0\
    --num_intent_clusters 10 \
    --cluster_value 0.3 \
    --intent_cf_weight 10\
    --cf_weight 0 \

## Amazon Toys

python main.py \
    --model_name UPTRec \
    --data_name Toys_and_Games  \
    --data_dir data \
    --context encoder \
    --seq_representation_type mean \
    --attention_type Cluster \
    --cluster_joint \
    --de_noise \
    --batch_size 256 \
    --epochs 2000 \
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 40 \
    --embedding \
    --output_dir Main_Table/Toys_and_Games/Item-Level \
    --model_idx V-Item-Level\
    --contrast_type Item-Level \
    --cluster_train 10 \
    --warm_up_epoches 0\
    --num_intent_clusters 10\
    --num_hidden_layers 2 \
    --cluster_value 0.3 \
    --intent_cf_weight 10\
    --cf_weight 0 \


## Yelp

python main.py \
    --model_name UPTRec \
    --data_name Yelp  \
    --data_dir data \
    --context encoder \
    --seq_representation_type mean \
    --attention_type Cluster \
    --cluster_joint \
    --de_noise \
    --batch_size 256 \
    --epochs 2000 \
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 40 \
    --embedding \
    --output_dir Main_Table/Yelp/Item-Level\
    --model_idx V-Item-Level\
    --contrast_type Item-Level \
    --cluster_train 10 \
    --warm_up_epoches 0\
    --num_intent_clusters 10 \
    --num_hidden_layers 2 \
    --cluster_value 0.3 \
    --intent_cf_weight 10\
    --cf_weight 0 \


# scripts/Beauty/Cluster_Attention_IntentCL_gpu_0.sh