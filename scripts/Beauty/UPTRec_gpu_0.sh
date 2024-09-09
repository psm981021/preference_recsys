
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
    --gpu_id 0 \
    --visualization_epoch 20 \
    --patience 30 \
    --embedding \
    --output_dir output/Beauty/Contrastive_Learning/User_level/V2_CL_User_prediction\
    --model_idx V2_CL_User_prediction\
    --contrast_type User \
    --augment_type mask \
    --n_views 3 \
    --cluster_train 1 \
    --warm_up_epoches 0\
    --cluster_attention_type 0 \
    --cluster_value 0.9 \
    --num_intent_clusters 10 \
    --intent_cf_weight 0.01 \
    --cf_weight 0.01 \
    --intent_cf_user_weight 0.01 \
    --num_hidden_layers 2 \
    --gamma 0.3 \
    --cluster_prediction \
    --temperature 1 \

# scripts/Beauty/UPTRec_gpu_0.sh