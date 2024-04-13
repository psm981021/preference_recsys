
python main.py --model_name="UPTRec" --data_name="Beauty"  --output_dir="output_custom/Beauty" \
    --model_idx="UPTRec_Baseline" \
    --contrast_type="None" --seq_representation_type="concatenate" \
    --num_intent_clusters=16 --gpu_id=0 \
    --epochs=3500  --patience=100\