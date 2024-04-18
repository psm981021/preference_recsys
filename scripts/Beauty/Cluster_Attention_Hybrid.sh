for augment_type in mask
do
    for gamma in 0.7
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Beauty"  \
            --output_dir="output_custom/Beauty" \
            --contrast_type="Hybrid" \
            --seq_representation_type="concatenate" \
            --attention_type="Cluster" \
            --model_idx="UPTRec_Clustered_Attention_Hybrid_${augment_type}_${gamma}" \
            --augment_type=$augment_type \
            --gamma=$gamma \
            --num_intent_clusters=16 --gpu_id=1 --epochs=1000 --patience=500 --warm_up_epoches=40 --de_noise

    done
done