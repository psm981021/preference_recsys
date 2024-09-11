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
    --output_dir output/Beauty/Cluster_Attention/V2_CL_17\
    --model_idx V2_CL_17\
    --contrast_type None \
    --cluster_train 1 \
    --warm_up_epoches 0\
    --num_intent_clusters 10 \
    --cluster_value 0.3 \

# scripts/Beauty/test.sh