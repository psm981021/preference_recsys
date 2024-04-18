
python main.py \
    --model_name="UPTRec" \
    --data_name="Beauty"  \
    --output_dir="output_custom/Beauty/Cluster_Attention" \
    --contrast_type="None" \
    --seq_representation_type="concatenate" \
    --attention_type="Cluster" \
    --batch_size=256\
    --model_idx="UPTRec_Clustered_Attention" \
    --num_intent_clusters=16 --gpu_id=0 --epochs=3500 --patience=500 --do_eval --attention_map \