for augment_type in mask
do
    for gamma in 0.7
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Beauty"  \
            --output_dir="output_custom/Beauty/Test/UPTRec_Clustered_Attention_Hybrid_test" \
            --contrast_type="Hybrid" \
            --context="item_embedding" \
            --seq_representation_type="concatenate" \
            --attention_type="Cluster" \
            --model_idx="UPTRec_Clustered_Attention__Hybrid_test" \
            --augment_type=$augment_type \
            --gamma=$gamma \
            --batch_size=512 \
            --gpu_id=1 \
            --n_views=2 \
            --epochs=10 \
            --temperature=0.5 \
            --patience=500 \
            --warm_up_epoches=0 \
            --num_intent_clusters=16 \
            --alignment_loss  \
            --attention_map \
            --embedding \
            --visualization_epoch=1 

    done
done


#scripts/Beauty/test.sh