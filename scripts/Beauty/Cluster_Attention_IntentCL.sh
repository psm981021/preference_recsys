for augment_type in mask
do
    for gamma in 0.7
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Beauty"  \
            --output_dir="output_custom/Beauty/Cluster_Attention_IntentCL" \
            --contrast_type="IntentCL" \
            --context="encoder" \
            --seq_representation_type="concatenate" \
            --attention_type="Cluster" \
            --model_idx="UPTRec_Clustered_Attention_IntentCL_${augment_type}_${gamma}_nviews_3_temperature_0.3" \
            --augment_type=$augment_type \
            --gamma=$gamma \
            --n_views=3\
            --num_intent_clusters=16 --temperature=0.3 --gpu_id=1 --epochs=1000 --patience=500 --warm_up_epoches=40 \

    done
done