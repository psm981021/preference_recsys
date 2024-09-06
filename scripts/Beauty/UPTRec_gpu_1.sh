
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
    --output_dir output/Beauty/Item_level/V_CL_48\
    --model_idx V_CL_48\
    --contrast_type None \
    --description \
    --align_weight 0.0001 \
    --augment_type mask \
    --n_views 3 \
    --cluster_train 1 \
    --warm_up_epoches 0\
    --num_intent_clusters 200 \
    --intent_cf_weight 0.01 \
    --cf_weight 1 \
    --num_hidden_layers 2 \
    --gamma 0.3 \
    --temperature 0.1 \

# scripts/Beauty/UPTRec_gpu_1.sh