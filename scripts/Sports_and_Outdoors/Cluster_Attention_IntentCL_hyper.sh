
for augment_type in mask
do
    for gamma in 0.3 0.5 0.7
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Sports_and_Outdoors"  \
            --output_dir="output_custom/Sports_and_Outdoors" \
            --contrast_type="IntentCL" \
            --seq_representation_type="concatenate" \
            --attention_type="Cluster" \
            --model_idx="UPTRec_Clustered_Attention_IntentCL_hyper_${augment_type}_${gamma}" \
            --augment_type=$augment_type \
            --gamma=$gamma \
            --num_intent_clusters=16 --gpu_id=0 --epochs=100 --patience=500\ 

    done
done

for augment_type in crop
do
    for tao in 0.1 0.2 0.3
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Sports_and_Outdoors"  \
            --output_dir="output_custom/Sports_and_Outdoors" \
            --contrast_type="IntentCL" \
            --seq_representation_type="concatenate" \
            --attention_type="Cluster" \
            --model_idx="UPTRec_Clustered_Attention_IntentCL_hyper_${augment_type}_${tao}" \
            --augment_type=$augment_type \
            --tao=$tao \
            --num_intent_clusters=16 --gpu_id=0 --epochs=100 --patience=500\ 

    done
done

for augment_type in reorder
do
    for beta in 0.1 0.2 0.3
    do
        python main.py \
            --model_name="UPTRec" \
            --data_name="Sports_and_Outdoors"  \
            --output_dir="output_custom/Sports_and_Outdoors" \
            --contrast_type="IntentCL" \
            --seq_representation_type="concatenate" \
            --attention_type="Cluster" \
            --model_idx="UPTRec_Clustered_Attention_IntentCL_hyper_${augment_type}_${beta}" \
            --augment_type=$augment_type \
            --beta=$beta \
            --num_intent_clusters=16 --gpu_id=0 --epochs=100 --patience=500\ 

    done
done