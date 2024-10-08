

for tao in 0.1 0.3 0.5 0.7
do
    python main.py \
        --model_name UPTRec \
        --data_name Beauty  \
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
        --output_dir Ablation/Beauty/Item_level/Augmentation/Mask_${tao} \
        --model_idx Mean\
        --augment_type mask \
        --tao $tao \
        --contrast_type Item-Level \
        --warm_up_epoches 0\
        --rec_weight 1 \
        --temperature 1 \
        --num_intent_clusters 10\
        --intent_cf_weight 1\
        --cf_weight 0 \
        --cluster_value 0.3 \
        --cluster_prediction
done

for beta in 0.1 0.2 0.3 0.4
do
    python main.py \
        --model_name UPTRec \
        --data_name Beauty  \
        --data_dir data/ \
        --context encoder \
        --seq_representation_type mean \
        --attention_type Cluster \
        --cluster_joint \
        --de_noise \
        --batch_size 256 \
        --epochs 2000 \
        --gpu_id 0 \
        --visualization_epoch 20 \
        --patience 30 \
        --embedding \
        --output_dir Ablation/Beauty/Item_level/Augmentation/Reorder_${beta} \
        --model_idx Mean\
        --augment_type mask \
        --beta $beta \
        --contrast_type Item-Level \
        --warm_up_epoches 0\
        --rec_weight 1 \
        --temperature 1 \
        --num_intent_clusters 10\
        --intent_cf_weight 1\
        --cf_weight 0 \
        --cluster_value 0.3 \
        --cluster_prediction
done


for beta in 0.1 0.2 0.3 0.4
do
    python main.py \
        --model_name UPTRec \
        --data_name Beauty  \
        --data_dir data/ \
        --context encoder \
        --seq_representation_type mean \
        --attention_type Cluster \
        --cluster_joint \
        --de_noise \
        --batch_size 256 \
        --epochs 2000 \
        --gpu_id 0 \
        --visualization_epoch 20 \
        --patience 30 \
        --embedding \
        --output_dir Ablation/Beauty/Item_level/Augmentation/Reorder_${beta} \
        --model_idx Mean\
        --augment_type mask \
        --beta $beta \
        --contrast_type Item-Level \
        --warm_up_epoches 0\
        --rec_weight 1 \
        --temperature 1 \
        --num_intent_clusters 10\
        --intent_cf_weight 1\
        --cf_weight 0 \
        --cluster_value 0.3 \
        --cluster_prediction
done

# scripts/Beauty/3.sh