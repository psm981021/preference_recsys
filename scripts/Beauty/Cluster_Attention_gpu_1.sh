python main.py \
    --model_name="UPTRec" \
    --data_name="Beauty"  \
    --output_dir="output_custom/Beauty/Cluster_Attention/UPTRec_Cluster_Attention_K(16)_R(200)_valid_self-attention" \
    --contrast_type="None" \
    --seq_representation_type="concatenate" \
    --attention_type="Cluster" \
    --batch_size=512 \
    --num_intent_clusters=16 \
    --epochs=4000 \
    --patience=500 \
    --context="item_embedding" \
    --gpu_id=1 \
    --cluster_train=200 \
    --visualization_epoch=100 \
    --model_idx="UPTRec_Cluster_Attention-200-valid-self-attention" \
    --rec_weight=1.5 \
    --alignment_loss \
    --embedding \
    --attention_map \
    --user_list=[] \
    --valid_attention \
    --wandb


# scripts/Beauty/Cluster_Attention_gpu_1.sh