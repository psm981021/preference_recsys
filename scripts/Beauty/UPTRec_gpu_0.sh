python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --output_dir output/Beauty/Item_level/V_CL_22\
    --contrast_type Item-level \
    --context item_embedding \
    --seq_representation_type concatenate \
    --attention_type Cluster \
    --batch_size 512 \
    --epochs 2000 \
    --patience 40 \
    --warm_up_epoches 50 \
    --num_intent_clusters 50 \
    --intent_cf_weight 1 \
    --cf_weight 0.1 \
    --num_hidden_layers 3 \
    --cluster_train 1 \
    --model_idx V_CL_22\
    --gpu_id 0 \
    --embedding \
    --n_views 3 \
    --visualization_epoch 50 \
    --temperature 0.8 \
    --cluster_joint \
    --augment_type mask\
    --de_noise

# scripts/Beauty/UPTRec_gpu_0.sh