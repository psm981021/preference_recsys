python main.py \
    --model_name="UPTRec" \
    --data_name="Beauty"  \
    --output_dir="output_custom/Beauty" \
    --contrast_type="IntentCL" \
    --seq_representation_type="concatenate" \
    --attention_type="Cluster" \
    --augment_type="mask" \
    --model_idx="UPTRec_Clustered_Attention_IntentCL" \
    --num_intent_clusters=16 --gpu_id=1 --epochs=4000 --patience=500 --de_noise\ 