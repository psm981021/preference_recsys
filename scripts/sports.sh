
for num_intent_clusters in 5 10 25
do
    for cluster_value in 0.3 0.5 0.7 1
    do
        python main.py \
            --model_name UPTRec \
            --data_name Sports_and_Outdoors  \
            --data_dir data/ \
            --context encoder \
            --seq_representation_type mean \
            --attention_type Cluster \
            --cluster_joint \
            --de_noise \
            --batch_size 256 \
            --epochs 2000 \
            --gpu_id 1 \
            --visualization_epoch 20 \
            --patience 30 \
            --embedding \
            --output_dir Main_Table/Sports/Item_level/${num_intent_clusters}_${cluster_value} \
            --model_idx Mean \
            --contrast_type Item-Level \
            --num_intent_clusters $num_intent_clusters\
            --cluster_value $cluster_value \
            --warm_up_epoches 0\
            --rec_weight 1 \
            --temperature 1 \
            --intent_cf_weight 0.5\
            --cf_weight 0.1
    done
done

# scripts/sports.sh