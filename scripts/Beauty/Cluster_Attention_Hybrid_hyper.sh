for augment_type in mask
do
    for gamma in 0.7
    do
        for n_views in 3 4
        do
            python main.py \
                --model_name="UPTRec" \
                --data_name="Beauty"  \
                --output_dir="output_custom/Beauty" \
                --contrast_type="Hybrid" \
                --seq_representation_type="concatenate" \
                --attention_type="Cluster" \
                --model_idx="UPTRec_Clustered_Attention_Hybrid_hyper_nviews_denoise(true)_${augment_type}_${gamma}_${n_views}" \
                --augment_type=$augment_type \
                --n_views=$n_views \
                --gamma=$gamma \
                --num_intent_clusters=16 --gpu_id=1 --epochs=100 --patience=500 --warm_up_epoches=10 --de_noise
        done
    done
done

for augment_type in mask
do
    for gamma in 0.7
    do
        for n_views in 2 4
        do
            python main.py \
                --model_name="UPTRec" \
                --data_name="Beauty"  \
                --output_dir="output_custom/Beauty" \
                --contrast_type="Hybrid" \
                --seq_representation_type="concatenate" \
                --attention_type="Cluster" \
                --model_idx="UPTRec_Clustered_Attention_Hybrid_hyper_nviews_denoise(false)${augment_type}_${gamma}_${n_views}" \
                --augment_type=$augment_type \
                --n_views=$n_views \
                --gamma=$gamma \
                --num_intent_clusters=16 --gpu_id=1 --epochs=100 --patience=500 --warm_up_epoches=10
        done
    done
done

for augment_type in mask
do
    for gamma in 0.7
    do
        for temperature in 0.5 1 2
        do
            python main.py \
                --model_name="UPTRec" \
                --data_name="Beauty"  \
                --output_dir="output_custom/Beauty" \
                --contrast_type="Hybrid" \
                --seq_representation_type="concatenate" \
                --attention_type="Cluster" \
                --model_idx="UPTRec_Clustered_Attention_Hybrid_hyper_temperature_${augment_type}_${gamma}_${temperature}" \
                --augment_type=$augment_type \
                --temperature=$temperature \
                --gamma=$gamma \
                --num_intent_clusters=16 --gpu_id=1 --epochs=100 --patience=500 --warm_up_epoches=10 --de_noise
        done
    done
done

for augment_type in mask
do
    for gamma in 0.7
    do
        for cf_weight in 0.01 0.02 0.05 
        do
            for rec_weight in 1.0 0.9 0.8 
            do
                for intent_cf_weight in 0.2 0.3 0.1
                do
                python main.py \
                    --model_name="UPTRec" \
                    --data_name="Beauty"  \
                    --output_dir="output_custom/Beauty" \
                    --contrast_type="Hybrid" \
                    --seq_representation_type="concatenate" \
                    --attention_type="Cluster" \
                    --model_idx="UPTRec_Clustered_Attention_Hybrid_hyper_weight_${augment_type}_${gamma}_${cf_weight}_${rec_weight}_${intent_cf_weight}" \
                    --augment_type=$augment_type \
                    --gamma=$gamma \
                    --num_intent_clusters=16 --gpu_id=1 --epochs=100 --patience=500 --warm_up_epoches=10 --de_noise 
        done
    done
done