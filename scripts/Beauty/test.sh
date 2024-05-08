for augment_type in mask
do
    for gamma in 0.7
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Beauty"  \
            --output_dir="output_custom/Beauty/Test/UPTRec_Clustered_Attention_test" \
            --contrast_type="None" \
            --context="item_embedding" \
            --seq_representation_type="concatenate" \
            --attention_type="Cluster" \
            --model_idx="UPTRec_Clustered_Attention_test" \
            --augment_type=$augment_type \
            --gamma=$gamma \
            --batch_size=512 \
            --gpu_id=1 \
            --n_views=2 \
            --epochs=200 \
            --temperature=0.5 \
            --patience=500 \
            --warm_up_epoches=100 \
            --num_intent_clusters=4 \
            --alignment_loss  \
            --embedding \
            --attention_map 

    done
done