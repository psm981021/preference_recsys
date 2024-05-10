
python main.py \
    --model_name="UPTRec" \
    --data_name="Beauty" \
    --output_dir="output_custom/Beauty/main_table" \
    --contrast_type="SASRec" \
    --seq_representation_type="concatenate" \
    --attention_type="Base" \
    --batch_size=512 \
    --epochs=4000 \
    --patience=500 \
    --gpu_id=0 \
    --attention_map \
    --wandb \
    --rec_weight=1.5 \
    --visualization_epoch=50 \
    --model_idx="UPTRec-Self-Attention_V2"


#output_custom/Beauty/main_table/baseline.sh