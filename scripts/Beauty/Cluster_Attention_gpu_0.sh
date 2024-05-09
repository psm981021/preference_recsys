
python main.py \
    --model_name="UPTRec" \
    --data_name="Beauty"  \
    --output_dir="output_custom/Beauty/Cluster_Attention/UPTRec_Cluster_Attention_50" \
    --contrast_type="None" \
    --seq_representation_type="concatenate" \
    --attention_type="Cluster" \
    --batch_size=512 \
    --num_intent_clusters=4 \
    --epochs=4000 \
    --patience=500 \
    --context="item_embedding" \
    --gpu_id=0 \
    --cluster_train=50 \
    --visualization_epoch=50 \
    --model_idx="UPTRec_Cluster_Attention-50" \
    --rec_weight=1.5 \
    --alignment_loss \
    --embedding \
    --user_list=[] \
    --attention_map \
    --wandb


# scripts/Beauty/Cluster_Attention_gpu_0.sh