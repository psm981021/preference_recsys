
python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --output_dir output/Beauty/Item_level/tests\
    --contrast_type Item-level \
    --context item_embedding \
    --seq_representation_type concatenate \
    --attention_type Cluster \
    --batch_size 512 \
    --epochs 2000 \
    --patience 40 \
    --warm_up_epoches 0 \
    --num_intent_clusters 5 \
    --intent_cf_weight 0.1 \
    --cf_weight 0.01 \
    --num_hidden_layers 2 \
    --cluster_train 1 \
    --model_idx tests\
    --gpu_id 0 \
    --embedding \
    --n_views 2 \
    --visualization_epoch 50 \
    --temperature 1 \
    --de_noise \

# scripts/Beauty/test.sh