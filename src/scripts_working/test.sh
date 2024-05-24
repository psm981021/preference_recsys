for num_cluster in 4 ; do

    for cluster_train in 1 ; do

        for n_views in 2 ; do

            for model_idx in Cluster-Attention-Vanilla-${num_cluster}-${cluster_train}-${n_views}; do

                python3 main.py \
                    --data_name Beauty \
                    --cf_weight 0.1 \
                    --output_dir UPTRec/${model_idx}/ \
                    --gpu_id 0 \
                    --batch_size 512 \
                    --model_idx model_idx \
                    --contrast_type Hybrid \
                    --seq_representation_type concatenate \
                    --cluster_attention \
                    --warm_up_epoches 0 \
                    --intent_cf_weight 0.1 \
                    --num_hidden_layers 1 \
                    --vanilla_attention \
                    --attention_map \
                    --num_intent_cluster ${num_cluster} \
                    --cluster_train ${cluster_train} \
                    --n_views ${n_views} 
            done
        done
    done
done

# scripts_working/test.sh