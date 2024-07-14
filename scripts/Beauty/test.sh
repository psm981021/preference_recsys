python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --output_dir Description/Beauty/test\
    --contrast_type Hybrid  \
    --context item_embedding \
    --seq_representation_type concatenate \
    --attention_type Base \
    --batch_size 512 \
    --epochs 2000 \
    --patience 40 \
    --warm_up_epoches 0 \
    --num_intent_clusters 32 \
    --intent_cf_weight 0.01 \
    --cf_weight 0.5 \
    --num_hidden_layers 1 \
    --cluster_train 1 \
    --model_idx V1 \
    --gpu_id 0 \
    --embedding \
    --description \
    --visualization_epoch 2 \
    --temperature 1 \


# scripts/Beauty/test.sh