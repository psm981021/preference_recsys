
# python main.py \
#     --model_name="UPTRec" \
#     --data_name="Beauty"  \
#     --output_dir="output_custom/Beauty/Cluster_Attention/UPTRec_Cluster_Attention_5" \
#     --contrast_type="None" \
#     --seq_representation_type="concatenate" \
#     --attention_type="Cluster" \
#     --batch_size=512 \
#     --num_intent_clusters=8 \
#     --epochs=2000 \
#     --patience=500 \
#     --context="item_embedding" \
#     --gpu_id=0\
#     --cluster_train=5 \
#     --model_idx="UPTRec_Cluster_Attention-5" \
#     --rec_weight=1.5 \
#     --alignment_loss \
#     --embedding \
#     --attention_map \
#     --wandb 
    


python main.py \
    --model_name="UPTRec" \
    --data_name="Beauty"  \
    --output_dir="output_custom/Beauty/Cluster_Attention/UPTRec_Cluster_Attention_20" \
    --contrast_type="None" \
    --seq_representation_type="concatenate" \
    --attention_type="Cluster" \
    --batch_size=512 \
    --num_intent_clusters=8 \
    --epochs=2000 \
    --patience=500 \
    --context="item_embedding" \
    --gpu_id=1 \
    --cluster_train=20 \
    --model_idx="UPTRec_Cluster_Attention-20" \
    --rec_weight=1.5 \
    --alignment_loss \
    --embedding \
    --attention_map \
    --wandb 


    # python main.py \
    # --model_name="UPTRec" \
    # --data_name="Beauty"  \
    # --output_dir="output_custom/Beauty/Cluster_Attention" \
    # --contrast_type="None" \
    # --seq_representation_type="concatenate" \
    # --attention_type="Cluster" \
    # --batch_size=512 \
    # --num_intent_clusters=8 \
    # --epochs=2000 \
    # --patience=500 \
    # --context="item_embedding" \
    # --gpu_id=1 \
    # --cluster_train=20 \
    # --model_idx="UPTRec_Cluster_Attention_20" \
    # --rec_weight=1.5 \
    # --alignment_loss \
    # --embedding \
    # --attention_map \
    # --wandb 


    # python main.py \
    # --model_name="UPTRec" \
    # --data_name="Beauty"  \
    # --output_dir="output_custom/Beauty/Cluster_Attention" \
    # --contrast_type="None" \
    # --seq_representation_type="concatenate" \
    # --attention_type="Cluster" \
    # --batch_size=512 \
    # --num_intent_clusters=4 \
    # --epochs=2000 \
    # --patience=500 \
    # --context="item_embedding" \
    # --gpu_id=1 \
    # --cluster_train=50 \
    # --model_idx="UPTRec_Cluster_Attention_50" \
    # --rec_weight=1.5 \
    # --alignment_loss \
    # --embedding \
    # --attention_map \
    # --wandb 