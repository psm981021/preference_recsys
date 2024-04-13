
python main.py --model_name="UPTRec" --data_name="Sports_and_Outdoors"  --output_dir="output_custom/Sports_and_Outdoors" \
    --model_idx="UPTRec_Clustered_Attention" \
    --contrast_type="None" --seq_representation_type="concatenate" \
    --num_intent_clusters=16  --attention_type="Cluster" \
    --epochs=3500 --gpu_id=1 \