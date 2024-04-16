for augment_type in reorder
do
    for beta in 0.2 
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Beauty"  \
            --output_dir="output_custom/Beauty" \
            --contrast_type="Hybrid" \
            --seq_representation_type="concatenate" \
            --attention_type="Cluster" \
            --model_idx="UPTRec_Clustered_Attention_Hybrid_${augment_type}_${beta}" \
            --augment_type=$augment_type \
            --beta=$beta \
            --num_intent_clusters=16 --gpu_id=1 --epochs=3000 --patience=200 --de_noise

    done
done