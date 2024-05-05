
python main.py \
    --model_name="UPTRec" \
    --data_name="Beauty" \
    --output_dir="output_custom/Beauty/main_table" \
    --contrast_type="SASRec" \
    --seq_representation_type="concatenate" \
    --attention_type="Base" \
    --batch_size=512 \
    --epochs=2000 \
    --patience=300 \
    --gpu_id=0 \
    --model_idx="UPTRec-Self-Attention"


#output_custom/Beauty/main_table/baseline.sh