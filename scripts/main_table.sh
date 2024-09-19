
## Amazon Sports


for contrast_type in IntentCL Hybrid Item-User Item-level User
do
    for cluster_value in 0.1 0.3 0.5
    do
        for cf_weight in 0 0.1 1 1.5
        do
            for rec_weight in 1 1.5 2
            do
                for intent_cf_weight in 0.01 0.1 1 2
                do
                    for num_intent_clusters in 5 10 25
                    do
                        for cluster_train in 1 5 10 20
                        do
                            for temperature in 0.1 1
                            do
                                for num_hidden_layers in 2 3
                                do
                                    python main.py \
                                        --model_name UPTRec \
                                        --data_name Sports_and_Outdoors  \
                                        --data_dir data/ \
                                        --context encoder \
                                        --seq_representation_type concatenate \
                                        --attention_type Cluster \
                                        --cluster_joint \
                                        --de_noise \
                                        --batch_size 512 \
                                        --epochs 2000 \
                                        --gpu_id 0 \
                                        --visualization_epoch 20 \
                                        --patience 40 \
                                        --embedding \
                                        --output_dir Main_Table/Amazon_Sports/${contrast_type}/V-${num_intent_clusters}-${cluster_train}-${cluster_value}-${cf_weight}-${rec_weight}-${intent_cf_weight}-${num_intent_clusters}-${temperature}-${num_hidden_layers}  \
                                        --model_idx V-${contrast_type}-${num_intent_clusters}-${cluster_train}\
                                        --contrast_type $contrast_type \
                                        --n_views 3 \
                                        --cluster_train $cluster_train \
                                        --warm_up_epoches 0\
                                        --num_intent_clusters $num_intent_clusters \
                                        --intent_cf_weight $intent_cf_weight \
                                        --cf_weight $cf_weight \
                                        --num_hidden_layers $num_hidden_layers \
                                        --cluster_value $cluster_value \
                                        --num_user_intent_clusters 256 \
                                        --temperature $temperature
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done


## ML-1M

for contrast_type in IntentCL Hybrid Item-User Item-level User
do
    for cluster_value in 0.1 0.3 0.5
    do
        for cf_weight in 0 0.1 1 1.5
        do
            for rec_weight in 1 1.5 2
            do
                for intent_cf_weight in 0.01 0.1 1 2
                do
                    for num_intent_clusters in 5 10 25 50
                    do
                        for cluster_train in 1 5 10 20
                        do
                            for temperature in 0.1 1
                            do
                                for num_hidden_layers in 2 3
                                do
                                    python main.py \
                                        --model_name UPTRec \
                                        --data_name ml-1m  \
                                        --data_dir data/ \
                                        --context encoder \
                                        --seq_representation_type concatenate \
                                        --attention_type Cluster \
                                        --cluster_joint \
                                        --de_noise \
                                        --batch_size 512 \
                                        --epochs 2000 \
                                        --gpu_id 0 \
                                        --visualization_epoch 20 \
                                        --patience 40 \
                                        --embedding \
                                        --max_seq_length 200 \
                                        --output_dir Main_Table/ml-1m/${contrast_type}/V-${num_intent_clusters}-${cluster_train}-${cluster_value}-${cf_weight}-${rec_weight}-${intent_cf_weight}-${num_intent_clusters}-${temperature}-${num_hidden_layers}  \
                                        --model_idx V-${num_intent_clusters}-${cluster_train}\
                                        --contrast_type $contrast_type \
                                        --n_views 3 \
                                        --cluster_train $cluster_train \
                                        --warm_up_epoches 0\
                                        --num_intent_clusters $num_intent_clusters \
                                        --intent_cf_weight $intent_cf_weight \
                                        --cf_weight $cf_weight \
                                        --num_hidden_layers $num_hidden_layers \
                                        --cluster_value $cluster_value \
                                        --num_user_intent_clusters 256 \
                                        --temperature $temperature
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

## Amazon Toys

for contrast_type in IntentCL Hybrid Item-User Item-level User
do
    for cluster_value in 0.1 0.3 0.5
    do
        for cf_weight in 0 0.1 1 1.5
        do
            for rec_weight in 1 1.5 2
            do
                for intent_cf_weight in 0.01 0.1 1 2
                do
                    for num_intent_clusters in 5 10 25
                    do
                        for cluster_train in 1 5 10 20
                        do
                            for temperature in 0.1 1
                            do
                                for num_hidden_layers in 2 3
                                do
                                    python main.py \
                                        --model_name UPTRec \
                                        --data_name Toys_and_Games  \
                                        --data_dir data/ \
                                        --context encoder \
                                        --seq_representation_type concatenate \
                                        --attention_type Cluster \
                                        --cluster_joint \
                                        --de_noise \
                                        --batch_size 512 \
                                        --epochs 2000 \
                                        --gpu_id 0 \
                                        --visualization_epoch 20 \
                                        --patience 40 \
                                        --embedding \
                                        --output_dir Main_Table/Toys_and_Games/${contrast_type}/V-${num_intent_clusters}-${cluster_train}-${cluster_value}-${cf_weight}-${rec_weight}-${intent_cf_weight}-${num_intent_clusters}-${temperature}-${num_hidden_layers}  \
                                        --model_idx V-${num_intent_clusters}-${cluster_train}\
                                        --contrast_type $contrast_type \
                                        --n_views 3 \
                                        --cluster_train $cluster_train \
                                        --warm_up_epoches 0\
                                        --num_intent_clusters $num_intent_clusters \
                                        --intent_cf_weight $intent_cf_weight \
                                        --cf_weight $cf_weight \
                                        --num_hidden_layers $num_hidden_layers \
                                        --cluster_value $cluster_value \
                                        --num_user_intent_clusters 256 \
                                        --temperature $temperature
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

## Yelp

for contrast_type in IntentCL Hybrid Item-User Item-level User
do
    for cluster_value in 0.1 0.3 0.5
    do
        for cf_weight in 0 0.1 1 1.5
        do
            for rec_weight in 1 1.5 2
            do
                for intent_cf_weight in 0.01 0.1 1 2
                do
                    for num_intent_clusters in 5 10 25
                    do
                        for cluster_train in 1 5 10 20
                        do
                            for temperature in 0.1 1
                            do
                                for num_hidden_layers in 2 3
                                do
                                    python main.py \
                                        --model_name UPTRec \
                                        --data_name Yelp  \
                                        --data_dir data/ \
                                        --context encoder \
                                        --seq_representation_type concatenate \
                                        --attention_type Cluster \
                                        --cluster_joint \
                                        --de_noise \
                                        --batch_size 512 \
                                        --epochs 2000 \
                                        --gpu_id 0 \
                                        --visualization_epoch 20 \
                                        --patience 40 \
                                        --embedding \
                                        --output_dir Main_Table/Yelp/${contrast_type}/V-${num_intent_clusters}-${cluster_train}-${cluster_value}-${cf_weight}-${rec_weight}-${intent_cf_weight}-${num_intent_clusters}-${temperature}-${num_hidden_layers}  \
                                        --model_idx V-${num_intent_clusters}-${cluster_train}\
                                        --contrast_type $contrast_type \
                                        --n_views 3 \
                                        --cluster_train $cluster_train \
                                        --warm_up_epoches 0\
                                        --num_intent_clusters $num_intent_clusters \
                                        --intent_cf_weight $intent_cf_weight \
                                        --cf_weight $cf_weight \
                                        --num_hidden_layers $num_hidden_layers \
                                        --cluster_value $cluster_value \
                                        --num_user_intent_clusters 256 \
                                        --temperature $temperature
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# ./scripts/main_table.sh