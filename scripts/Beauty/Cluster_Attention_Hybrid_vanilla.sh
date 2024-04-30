for augment_type in mask
do
    for gamma in 0.7
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Beauty"  \
            --output_dir="output_custom/Beauty/Cluster_Attention_Hybrid" \
            --contrast_type="Hybrid" \
            --context="item_embedding" \
            --seq_representation_type="concatenate" \
            --attention_type="Cluster" \
            --augment_type=$augment_type \
            --gamma=$gamma \
            --batch_size=512 \
            --gpu_id=1 \
            --n_views=3 \
            --epochs=2000 \
            --temperature=0.1 \
            --patience=500 \
            --warm_up_epoches=200 \
            --num_intent_clusters=64 \
            --vanilla_attention \
            --de_noise \
            --alignment_loss \ 
            --model_idx="UPTRec_Clustered_Attention_Hybrid_${augment_type}_${gamma}_encoder_vanilla"

    done
done