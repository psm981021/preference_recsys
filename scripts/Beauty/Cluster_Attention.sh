
python main.py \
    --model_name="UPTRec" \
    --data_name="Beauty"  \
    --output_dir="output_custom/Beauty" \
    --contrast_type="None" \
    --seq_representation_type="concatenate" \
    --attention_type="Cluster" \
    --batch_size=512\
    --model_idx="UPTRec_Clustered_Attention_valid_test_adjusted" \
    --num_intent_clusters=32 --gpu_id=0 --epochs=3500 --patience=500\ 