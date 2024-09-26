
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
    --output_dir Ablation/Beauty/Item_level/6/Mean \
    --model_idx Mean\
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 10\
    --intent_cf_weight 2\
    --cf_weight 0 \
    --cluster_value 0 \

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
    --output_dir Ablation/Beauty/Item_level/6/Mean_NCL \
    --model_idx Mean_NCL\
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 10\
    --intent_cf_weight 2\
    --cf_weight 0 \
    --cluster_value 0 \
    --ncl \

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
    --output_dir Ablation/Beauty/Item_level/6/Mean_temperature_density \
    --model_idx Mean_temperature_density\
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 10\
    --intent_cf_weight 2\
    --cf_weight 0 \
    --cluster_value 0 \
    --cluster_temperature \

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
    --output_dir Ablation/Beauty/Item_level/6/Mean_Prediction \
    --model_idx Mean_Prediction\
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 10\
    --intent_cf_weight 2\
    --cf_weight 0 \
    --cluster_value 0 \
    --cluster_prediction \

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
    --output_dir Ablation/Beauty/Item_level/6/Mean_MLP \
    --model_idx Mean_MLP\
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 10\
    --intent_cf_weight 2\
    --cf_weight 0 \
    --cluster_value 0 \
    --mlp \

# ## Amazon Sports

# python main.py \
#     --model_name UPTRec \
#     --data_name Sports_and_Outdoors  \
#     --data_dir data/ \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 512 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Main_Table/Amazon_Sports/Cluster_Attention  \
#     --model_idx V-Cluster_Attention\
#     --contrast_type None \
#     --cluster_train 10 \
#     --warm_up_epoches 0\
#     --num_intent_clusters 10 \
#     --num_hidden_layers 2 \
#     --cluster_value 0.3 \


# ## ML-1M

# python main.py \
#     --model_name UPTRec \
#     --data_name ml-1m  \
#     --data_dir data \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 512 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --max_seq_length 200 \
#     --output_dir Main_Table/ml-1m/Cluster_Attention \
#     --model_idx V-Cluster_Attention\
#     --contrast_type None \
#     --cluster_train 10 \
#     --num_hidden_layers 2 \
#     --warm_up_epoches 0\
#     --num_intent_clusters 10 \
#     --cluster_value 0.3 \
#     --do_eval \

# ## Amazon Toys

# python main.py \
#     --model_name UPTRec \
#     --data_name Toys_and_Games  \
#     --data_dir data \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 512 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Main_Table/Toys_and_Games/Cluster_Attention \
#     --model_idx V-Cluster_Attention\
#     --contrast_type None \
#     --cluster_train 10 \
#     --warm_up_epoches 0\
#     --num_intent_clusters 10\
#     --num_hidden_layers 2 \
#     --cluster_value 0.3 \
#     --do_eval \


# ## Yelp

# python main.py \
#     --model_name UPTRec \
#     --data_name Yelp  \
#     --data_dir data \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 512 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Main_Table/Yelp/Cluster_Attention\
#     --model_idx V-Cluster_Attention\
#     --contrast_type None \
#     --cluster_train 10 \
#     --warm_up_epoches 0\
#     --num_intent_clusters 10 \
#     --num_hidden_layers 2 \
#     --cluster_value 0.3 \
#     --do_eval \



# ./scripts/main_table.sh