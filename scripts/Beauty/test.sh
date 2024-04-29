for augment_type in mask
do
    for gamma in 0.7
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Beauty"  \
            --output_dir="output_custom/Beauty/" \
            --contrast_type="Hybrid" \
            --context="item_embedding" \
            --seq_representation_type="concatenate" \
            --attention_type="Cluster" \
            --model_idx="UPTRec_Clustered_Attention_IntentCL_${augment_type}_${gamma}_nviews_3_test" \
            --augment_type=$augment_type \
            --gamma=$gamma \
            --gpu_id=1 \
            --n_views=4 \
            --epochs=20 \
            --temperature=0.5 \
            --patience=500 \
            --warm_up_epoches=0 \
            --num_intent_clusters=16 \
            --alignment_loss
    done
done