for augment_type in mask
do
    for gamma in 0.7
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Beauty"  \
            --output_dir="output_custom/Beauty/Cluster_Attention_IntentCL/K(16)-R(200)" \
            --contrast_type="IntentCL" \
            --context="item_embedding" \
            --seq_representation_type="concatenate" \
            --attention_type="Cluster" \
            --augment_type=$augment_type \
            --gamma=$gamma \
            --batch_size=512 \
            --gpu_id=1 \
            --n_views=3 \
            --rec_weight=1.5 \
            --epochs=4000 \
            --intent_cf_weight=0.5 \
            --temperature=0.1 \
            --patience=500 \
            --embedding \
            --attention_map \
            --warm_up_epoches=200 \
            --num_intent_clusters=16 \
            --visualization_epoch=100 \
            --cluster_train=200 \
            --user_list=[] \
            --wandb \
            --model_idx="UPTRec_Clustered_Attention_IntentCL_${augment_type}_${gamma}_K(16)_R(200)"

    done
done