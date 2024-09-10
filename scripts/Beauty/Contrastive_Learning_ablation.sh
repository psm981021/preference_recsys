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
    --output_dir output/Beauty/Contrastive_Learning/Item_level/V2_CL_01 \
    --model_idx V2_CL_01\
    --contrast_type Item-level \
    --augment_type random \
    --gamma 0.3 \
    --n_views 3 \
    --num_intent_clusters 10 \
    --cluster_train 1 \
    --warm_up_epoches 0\
    --intent_cf_weight 0.1 \
    --cf_weight 1 \
    --cluster_value 0.7 \
    --intent_cf_user_weight 0.1 \
    --num_hidden_layers 2 \
    --temperature 1

# python main.py \
#     --model_name UPTRec \
#     --data_name Beauty  \
#     --context encoder \
#     --seq_representation_type concatenate \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 512 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir output/Beauty/Contrastive_Learning/User_level/V2_CL_02 \
#     --model_idx V2_CL_02\
#     --contrast_type User \
#     --augment_type random \
#     --gamma 0.3 \
#     --n_views 3 \
#     --cluster_train 1 \
#     --warm_up_epoches 0\
#     --num_intent_clusters 10 \
#     --intent_cf_weight 0.01 \
#     --cf_weight 0.01 \
#     --cluster_value 0.7 \
#     --intent_cf_user_weight 0.01 \
#     --num_hidden_layers 2 \
#     --temperature 1

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
    --output_dir output/Beauty/Contrastive_Learning/Item_User_level/V2_CL_03 \
    --model_idx V2_CL_03\
    --contrast_type Item-User \
    --augment_type random \
    --gamma 0.3 \
    --n_views 3 \
    --num_intent_clusters 10 \
    --cluster_train 1 \
    --warm_up_epoches 0\
    --intent_cf_weight 1 \
    --cf_weight 0.1 \
    --intent_cf_user_weight 0.01 \
    --cluster_value 0.8 \
    --num_hidden_layers 2 \
    --temperature 0.01

# python main.py \
#     --model_name UPTRec \
#     --data_name Beauty  \
#     --context encoder \
#     --seq_representation_type concatenate \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 512 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir output/Beauty/Contrastive_Learning/Item_description_level/V2_CL_02 \
#     --model_idx V2_CL_02\
#     --contrast_type Item-description \
#     --augment_type random \
#     --gamma 0.3 \
#     --n_views 3 \
#     --cluster_train 1 \
#     --num_intent_clusters 10 \
#     --warm_up_epoches 0\
#     --align_weight 0.001 \
#     --intent_cf_weight 0.01 \
#     --description \
#     --cf_weight 1 \
#     --intent_cf_user_weight 0.01 \
#     --cluster_value 0.7 \
#     --num_hidden_layers 2 \
#     --temperature 1


# scripts/Beauty/Contrastive_Learning_ablation.sh