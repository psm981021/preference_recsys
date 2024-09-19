
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
    --output_dir output/Beauty/Contrastive_Learning/User_level/V2_CL_06 \
    --model_idx V2_CL_06\
    --contrast_type User \
    --cluster_train 10 \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 0.1 \
    --num_intent_clusters 10\
    --intent_cf_weight 0.1\
    --cf_weight 1 \
    --cluster_value 0.7 \

# scripts/Beauty/UPTRec_gpu_1.sh