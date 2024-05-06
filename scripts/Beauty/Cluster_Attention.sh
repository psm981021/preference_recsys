
# python main.py \
#     --model_name="UPTRec" \
#     --data_name="Beauty"  \
#     --output_dir="output_custom/Beauty/Cluster_Attention" \
#     --contrast_type="None" \
#     --seq_representation_type="concatenate" \
#     --attention_type="Cluster" \
#     --batch_size=512 \
#     --num_intent_clusters=64 \
#     --epochs=2000 \
#     --patience=500 \
#     --context="item_embedding" \
#     --gpu_id=0\
#     --cluster_train=5 \
#     --model_idx="UPTRec_Cluster_Attention_5" \
#     --alignment_loss 
    


# python main.py \
#     --model_name="UPTRec" \
#     --data_name="Beauty"  \
#     --output_dir="output_custom/Beauty/Cluster_Attention" \
#     --contrast_type="None" \
#     --seq_representation_type="concatenate" \
#     --attention_type="Cluster" \
#     --batch_size=512 \
#     --num_intent_clusters=64 \
#     --epochs=2000 \
#     --patience=500 \
#     --context="item_embedding" \
#     --gpu_id=1 \
#     --cluster_train=10 \
#     --model_idx="UPTRec_Cluster_Attention_10" \
#     --alignment_loss 

    python main.py \
    --model_name="UPTRec" \
    --data_name="Beauty"  \
    --output_dir="output_custom/Beauty/Cluster_Attention" \
    --contrast_type="None" \
    --seq_representation_type="concatenate" \
    --attention_type="Cluster" \
    --batch_size=512 \
    --num_intent_clusters=64 \
    --epochs=2000 \
    --patience=500 \
    --context="item_embedding" \
    --gpu_id=1 \
    --cluster_train=20 \
    --model_idx="UPTRec_Cluster_Attention_20" \
    --alignment_loss 

    python main.py \
    --model_name="UPTRec" \
    --data_name="Beauty"  \
    --output_dir="output_custom/Beauty/Cluster_Attention" \
    --contrast_type="None" \
    --seq_representation_type="concatenate" \
    --attention_type="Cluster" \
    --batch_size=512 \
    --num_intent_clusters=64 \
    --epochs=2000 \
    --patience=500 \
    --context="item_embedding" \
    --gpu_id=1 \
    --cluster_train=50 \
    --model_idx="UPTRec_Cluster_Attention_50" \
    --alignment_loss 