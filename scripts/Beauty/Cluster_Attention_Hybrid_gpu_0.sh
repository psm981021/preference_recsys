for augment_type in random
do
    for gamma in 0.7
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Beauty"  \
            --output_dir="renewlog/Beauty/Cluster_Attention_Hybrid/Hybrid-K(16)-R(200)_V3" \
            --contrast_type="Hybrid" \
            --context="item_embedding" \
            --seq_representation_type="mean" \
            --attention_type="Cluster" \
            --augment_type=$augment_type \
            --gamma=$gamma \
            --batch_size=512 \
            --rec_weight=1.5 \
            --gpu_id=0 \
            --n_views=3 \
            --epochs=500 \
            --intent_cf_weight=0.1 \
            --num_hidden_layers=1 \
            --temperature=0.1 \
            --patience=40 \
            --warm_up_epoches=0 \
            --num_intent_clusters=16 \
            --cluster_train=1 \
            --visualization_epoch=100 \
            --alignment_loss \
            --embedding \
            --attention_map \
            --user_list=[] \
            --model_idx="UPTRec_Clustered_Attention_Hybrid_${augment_type}_V3"
    done
done

#scripts/Beauty/Cluster_Attention_Hybrid_gpu_0.sh