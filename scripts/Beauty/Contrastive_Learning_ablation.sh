for num_intent_clusters in 5 10 25 50 100 200 500
do
    for cluster_value in 1 0.9 0.7 0.5 0.2 0.1 
    do
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
        --output_dir output/Beauty/Abation/Item_level_Contrastive_learning/CL_${num_intent_clusters}_${cluster_value}\
        --model_idx CA_${num_intent_clusters}_${cluster_value}\
        --contrast_type Item-level \
        --augment_type random \
        --n_views 3 \
        --cluster_train 1 \
        --warm_up_epoches 0\
        --intent_cf_weight 0.01 \
        --cf_weight 1 \
        --intent_cf_user_weight 0.1 \
        --num_hidden_layers 2 \
        --temperature 1
    done
done


for num_intent_clusters in 5 10 25 50 100 200 500
do
    for cluster_value in 1 0.9 0.7 0.5 0.2 0.1 
    do
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
        --output_dir output/Beauty/Abation/Item-User_Contrastive_learning/CL_${num_intent_clusters}_${cluster_value}\
        --model_idx CA_${num_intent_clusters}_${cluster_value}\
        --contrast_type Item-User \
        --augment_type random \
        --n_views 3 \
        --cluster_train 1 \
        --warm_up_epoches 0\
        --intent_cf_weight 0.01 \
        --cf_weight 1 \
        --intent_cf_user_weight 0.1 \
        --num_hidden_layers 2 \
        --temperature 1
    done
done

for num_intent_clusters in 5 10 25 50 100 200 500
do
    for cluster_value in 1 0.9 0.7 0.5 0.2 0.1 
    do
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
        --output_dir output/Beauty/Abation/User_Contrastive_learning/CL_${num_intent_clusters}_${cluster_value}\
        --model_idx CA_${num_intent_clusters}_${cluster_value}\
        --contrast_type User \
        --augment_type random \
        --n_views 3 \
        --cluster_train 1 \
        --warm_up_epoches 0\
        --intent_cf_weight 0.1 \
        --cf_weight 1 \
        --intent_cf_user_weight 0.01 \
        --num_hidden_layers 2 \
        --temperature 1
    done
done

# scripts/Beauty/Contrastive_Learning_ablation.sh