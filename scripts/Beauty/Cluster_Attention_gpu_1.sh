python main.py \
    --model_name="UPTRec" \
    --data_name="Beauty"  \
    --output_dir="output_custom/Beauty/Cluster_Attention/UPTRec_Cluster_Attention_100" \
    --contrast_type="None" \
    --seq_representation_type="concatenate" \
    --attention_type="Cluster" \
    --batch_size=512 \
    --num_intent_clusters=8 \
    --epochs=2000 \
    --patience=500 \
    --context="item_embedding" \
    --gpu_id=1 \
    --cluster_train=100 \
    --model_idx="UPTRec_Cluster_Attention-100" \
    --rec_weight=1.5 \
    --alignment_loss \
    --embedding \
    --attention_map \
    --wandb


# scripts/Beauty/Cluster_Attention_gpu_1.sh