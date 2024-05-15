python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --output_dir renew_log/Beauty/UPTRec-16-encoder-concatenate-cluster_valid/\
    --contrast_type Hybrid  \
    --context encoder \
    --seq_representation_type concatenate \
    --attention_type Cluster \
    --batch_size 512 \
    --epochs 2000 \
    --patience 50 \
    --warm_up_epoches 0 \
    --num_intent_clusters 16 \
    --intent_cf_weight 0.1 \
    --cf_weight 0.1 \
    --num_hidden_layers 1 \
    --model_idx UPTRec-encoder-concatenate-cluster_valid \
    --gpu_id 1 \
    --embedding \
    --visualization_epoch 10 \
    --temperature 1 \
    --cluster_valid \
    --wandb

python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --output_dir renew_log/Beauty/UPTRec-16-encoder-concatenate-cluster_train_10/\
    --contrast_type Hybrid  \
    --context encoder \
    --seq_representation_type concatenate \
    --attention_type Cluster \
    --batch_size 512 \
    --epochs 2000 \
    --patience 50 \
    --warm_up_epoches 0 \
    --num_intent_clusters 16 \
    --intent_cf_weight 0.1 \
    --cf_weight 0.1 \
    --num_hidden_layers 1 \
    --model_idx UPTRec-encoder-concatenate-cluster_train_10 \
    --gpu_id 1 \
    --embedding \
    --visualization_epoch 10 \
    --temperature 1 \
    --wandb

python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --output_dir renew_log/Beauty/UPTRec-16-1-encoder-concatenate-hidden_2/\
    --contrast_type Hybrid  \
    --context encoder \
    --seq_representation_type concatenate \
    --attention_type Cluster \
    --batch_size 512 \
    --epochs 2000 \
    --patience 50 \
    --warm_up_epoches 0 \
    --num_intent_clusters 16 \
    --intent_cf_weight 0.1 \
    --cf_weight 0.1 \
    --num_hidden_layers 2 \
    --model_idx UPTRec-encoder-concatenate-hidden_2 \
    --gpu_id 1 \
    --embedding \
    --visualization_epoch 10 \
    --temperature 1 \
    --wandb


python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --output_dir renew_log/Beauty/UPTRec-16-item_embedding-concatenate/\
    --contrast_type Hybrid  \
    --context item_embedding \
    --seq_representation_type concatenate \
    --attention_type Cluster \
    --batch_size 512 \
    --epochs 2000 \
    --patience 50 \
    --warm_up_epoches 0 \
    --num_intent_clusters 16 \
    --intent_cf_weight 0.1 \
    --cf_weight 0.1 \
    --num_hidden_layers 1 \
    --model_idx UPTRec-item_embedding-concatenate \
    --gpu_id 1 \
    --embedding \
    --visualization_epoch 10 \
    --temperature 1 \
    --wandb

python main.py \
    --model_name UPTRec \
    --data_name Beauty  \
    --output_dir renew_log/Beauty/UPTRec-16-0.5-encoder-concatenate-cluster_valid/\
    --contrast_type Hybrid  \
    --context encoder \
    --seq_representation_type concatenate \
    --attention_type Cluster \
    --batch_size 512 \
    --epochs 2000 \
    --patience 50 \
    --warm_up_epoches 0 \
    --num_intent_clusters 16 \
    --intent_cf_weight 0.1 \
    --cf_weight 0.1 \
    --num_hidden_layers 1 \
    --model_idx UPTRec-item_embedding-concatenate-cluster_valid \
    --gpu_id 1 \
    --embedding \
    --visualization_epoch 10 \
    --temperature 0.5 \
    --cluster_valid \
    --wandb

# scripts/Beauty/UPTRec_gpu_1.sh