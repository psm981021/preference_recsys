for augment_type in mask
do
    for gamma in 0.7
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Beauty"  \
            --output_dir="output_custom/Beauty/Cluster_Attention_IntentCL" \
            --contrast_type="IntentCL" \
            --context="item_embedding" \
            --seq_representation_type="concatenate" \
            --attention_type="Cluster" \
            --model_idx="UPTRec_Clustered_Attention_IntentCL_${augment_type}_${gamma}_test" \
            --augment_type=$augment_type \
            --gamma=$gamma \
            --gpu_id=0 \
            --n_views=3 \
            --epochs=1000 \
            --temperature=0.5 \
            --patience=500 \
            --warm_up_epoches=0 \
            --num_intent_clusters=16 \

    done
done