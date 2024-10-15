python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
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
<<<<<<< HEAD
    --output_dir Ablation/Beauty/Item_level/Invariant_Aug/Base\
=======
    --output_dir Ablation/Beauty/Item_level/35 \
>>>>>>> f1756f202471f907d4f4607a03414112ea31c4c4
    --model_idx Mean \
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 10\
<<<<<<< HEAD
    --intent_cf_weight 1 \
    --cf_weight 1 \
    --cluster_value 0.3 \
    --invariant_augment \
    --num_hidden_layers 3 \
    --pre_train \
=======
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

>>>>>>> f1756f202471f907d4f4607a03414112ea31c4c4

python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
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
    --output_dir Ablation/Beauty/Item_level/Invariant_Aug/Bi-direction\
    --model_idx Mean \
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 10\
    --intent_cf_weight 1 \
    --cf_weight 1 \
    --cluster_value 0.3 \
    --invariant_augment \
    --num_hidden_layers 3 \
    --bi_direction \
    --pre_train \

python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
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
    --output_dir Ablation/Beauty/Item_level/Invariant_Aug/NCL\
    --model_idx Mean \
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 10\
    --intent_cf_weight 1 \
    --cf_weight 1 \
    --cluster_value 0.3 \
    --invariant_augment \
    --num_hidden_layers 3 \
    --ncl \
    --pre_train \

python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
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
    --output_dir Ablation/Beauty/Item_level/Invariant_Aug/Cluster_tempearture\
    --model_idx Mean \
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 10\
    --intent_cf_weight 1 \
    --cf_weight 1 \
    --cluster_value 0.3 \
    --invariant_augment \
    --num_hidden_layers 3 \
    --cluster_temperature \
    --pre_train \


# ./scripts/Beauty/test.sh