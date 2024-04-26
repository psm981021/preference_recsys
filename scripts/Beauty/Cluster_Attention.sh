
# python main.py \
#     --model_name="UPTRec" \
#     --data_name="Beauty"  \
#     --output_dir="output_custom/Beauty/Cluster_Attention" \
#     --contrast_type="None" \
#     --seq_representation_type="concatenate" \
#     --attention_type="Cluster" \
#     --batch_size=256 \
#     --model_idx="UPTRec_Cluster_Attention_vanilla_attention" \
#     --num_intent_clusters=16 \
#     --warm_up_epoches=150 \
#     --epochs=1000 \
#     --patience=500 \
#     --vanilla_attention \
#     --context="item_embedding" \
#     --gpu_id=0 


python main.py \
    --model_name="UPTRec" \
    --data_name="Beauty"  \
    --output_dir="output_custom/Beauty/Cluster_Attention" \
    --contrast_type="None" \
    --seq_representation_type="concatenate" \
    --attention_type="Cluster" \
    --batch_size=256 \
    --model_idx="UPTRec_Cluster_Attention_self_attention" \
    --num_intent_clusters=16 \
    --warm_up_epoches=150 \
    --epochs=1500 \
    --patience=500 \
    --context="item_embedding" \
    --gpu_id=1