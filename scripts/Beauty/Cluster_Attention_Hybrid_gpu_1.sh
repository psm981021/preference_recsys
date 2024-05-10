for augment_type in mask
do
    for gamma in 0.7
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Beauty"  \
            --output_dir="output_custom/Beauty/Cluster_Attention_Hybrid/Hybrid-K(16)-R(200)" \
            --contrast_type="Hybrid" \
            --context="item_embedding" \
            --seq_representation_type="concatenate" \
            --attention_type="Cluster" \
            --augment_type=$augment_type \
            --gamma=$gamma \
            --batch_size=512 \
            --rec_weight=1.5 \
            --gpu_id=1 \
            --n_views=3 \
            --epochs=4000 \
            --intent_cf_weight=0.5 \
            --temperature=0.1 \
            --patience=500 \
            --warm_up_epoches=300 \
            --num_intent_clusters=16 \
            --cluster_train=100 \
            --visualization_epoch=100 \
            --alignment_loss \
            --embedding \
            --attention_map \
            --user_list=[] \
            --de_noise \
            --model_idx="UPTRec_Clustered_Attention_Hybrid_${augment_type}_${gamma}_" \
            --wandb
    done
done

#scripts/Beauty/Cluster_Attention_Hybrid_gpu_1.sh