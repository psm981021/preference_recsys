
python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --context encoder \
    --seq_representation_type concatenate \
    --attention_type Cluster \
    --cluster_joint \
    --de_noise \
    --batch_size 256 \
    --epochs 2000 \
    --gpu_id 1 \
    --visualization_epoch 50 \
    --patience 40 \
    --embedding \
    --output_dir output/Beauty/Item_level/V_CL_39\
    --model_idx V_CL_39\
    --contrast_type IntentCL \
    --augment_type random \
    --n_views 3 \
    --cluster_train 10 \
    --warm_up_epoches 0\
    --num_intent_clusters 5 \
    --intent_cf_weight 0.1 \
    --cf_weight 0.1 \
    --lr 0.01 \
    --hidden_size 128 \
    --num_hidden_layers 3 \
    --temperature 0.1 \

# scripts/Beauty/UPTRec_gpu_1.sh