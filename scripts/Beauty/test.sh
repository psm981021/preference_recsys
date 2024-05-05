for augment_type in mask
do
    for gamma in 0.7
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Beauty"  \
            --output_dir="output_custom/Beauty/" \
            --contrast_type="IntentCL" \
            --context="item_embedding" \
            --seq_representation_type="concatenate" \
            --attention_type="Cluster" \
            --model_idx="UPTRec_Clustered_Attention_IntentCL_test" \
            --augment_type=$augment_type \
            --gamma=$gamma \
            --batch_size=512 \
            --gpu_id=0 \
            --n_views=2 \
            --epochs=200 \
            --temperature=0.5 \
            --patience=500 \
            --warm_up_epoches=100 \
            --num_intent_clusters=64 \
            --alignment_loss 
    done
done