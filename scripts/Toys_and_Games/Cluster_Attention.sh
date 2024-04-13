
python main.py --model_name="UPTRec" --data_name="Toys_and_Games"  --output_dir="output_custom/Toys_and_Games" \
    --model_idx="UPTRec_Clustered_Attention" \
    --contrast_type="None" --seq_representation_type="concatenate" \
    --num_intent_clusters=16 --gpu_id=0 --attention_type="Cluster" \
    --epochs=3500 \