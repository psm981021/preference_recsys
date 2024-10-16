# python main.py \
#     --model_name CLARRec \
#     --data_name Baby  \
#     --data_dir data/10core \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Main_Table/10core/K:10/Baby/ \
#     --model_idx Mean\
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 1 \
#     --num_intent_clusters 10\
#     --intent_cf_weight 1.2\
#     --cf_weight 0 \
#     --cluster_value 0.3\


# python main.py \
#     --model_name CLARRec \
#     --data_name Beauty  \
#     --data_dir data/10core \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Main_Table/10core/K:10/Beauty/ \
#     --model_idx Mean\
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 1 \
#     --num_intent_clusters 10\
#     --intent_cf_weight 1.2\
#     --cf_weight 0 \
#     --cluster_value 0.3\

python main.py \
    --model_name CLARRec \
    --data_name Clothing_Shoes_and_Jewelry  \
    --data_dir data/10core \
    --context encoder \
    --seq_representation_type mean \
    --attention_type Cluster \
    --cluster_joint \
    --de_noise \
    --batch_size 256 \
    --epochs 2000 \
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 30 \
    --embedding \
    --output_dir Main_Table/10core/K:10/Clothing_Shoes_and_Jewelry/ \
    --model_idx Mean\
    --contrast_type Item-Level \
    --warm_up_epoches 0\
    --rec_weight 1 \
    --temperature 1 \
    --num_intent_clusters 10\
    --intent_cf_weight 1.2\
    --cf_weight 0 \
    --cluster_value 0.3\

# python main.py \
#     --model_name CLARRec \
#     --data_name Electronics  \
#     --data_dir data/10core \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Main_Table/10core/K:10/Electronics/ \
#     --model_idx Mean\
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 1 \
#     --num_intent_clusters 10\
#     --intent_cf_weight 1.2\
#     --cf_weight 0 \
#     --cluster_value 0.3\

# python main.py \
#     --model_name CLARRec \
#     --data_name Grocery_and_Gourmet_Food  \
#     --data_dir data/10core \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Main_Table/10core/K:10/Grocery_and_Gourmet_Food/ \
#     --model_idx Mean\
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 1 \
#     --num_intent_clusters 10\
#     --intent_cf_weight 1.2\
#     --cf_weight 0 \
#     --cluster_value 0.3\


# python main.py \
#     --model_name CLARRec \
#     --data_name Home_and_Kitchen  \
#     --data_dir data/10core \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Main_Table/10core/K:10/Home_and_Kitchen/ \
#     --model_idx Mean\
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 1 \
#     --num_intent_clusters 10\
#     --intent_cf_weight 1.2\
#     --cf_weight 0 \
#     --cluster_value 0.3\

# python main.py \
#     --model_name CLARRec \
#     --data_name Sports_and_Outdoors  \
#     --data_dir data/10core \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Main_Table/10core/K:10/Sports_and_Outdoors/ \
#     --model_idx Mean\
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 1 \
#     --num_intent_clusters 10\
#     --intent_cf_weight 1.2\
#     --cf_weight 0 \
#     --cluster_value 0.3\


# python main.py \
#     --model_name CLARRec \
#     --data_name Tools_and_Home_Improvement  \
#     --data_dir data/10core \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Main_Table/10core/K:10/Tools_and_Home_Improvement/ \
#     --model_idx Mean\
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 1 \
#     --num_intent_clusters 10\
#     --intent_cf_weight 1.2\
#     --cf_weight 0 \
#     --cluster_value 0.3\

# python main.py \
#     --model_name CLARRec \
#     --data_name Toys_and_Games  \
#     --data_dir data/10core \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Main_Table/10core/K:10/Toys_and_Games/ \
#     --model_idx Mean\
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 1 \
#     --num_intent_clusters 10\
#     --intent_cf_weight 1.2\
#     --cf_weight 0 \
#     --cluster_value 0.3\


# python main.py \
#     --model_name CLARRec \
#     --data_name Video_Games  \
#     --data_dir data/10core \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Main_Table/10core/K:10/Video_Games/ \
#     --model_idx Mean\
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 1 \
#     --num_intent_clusters 10\
#     --intent_cf_weight 1.2\
#     --cf_weight 0 \
#     --cluster_value 0.3\

# python main.py \
#     --model_name CLARRec \
#     --data_name Movies_and_TV  \
#     --data_dir data/10core \
#     --context encoder \
#     --seq_representation_type mean \
#     --attention_type Cluster \
#     --cluster_joint \
#     --de_noise \
#     --batch_size 256 \
#     --epochs 2000 \
#     --gpu_id 1 \
#     --visualization_epoch 20 \
#     --patience 30 \
#     --embedding \
#     --output_dir Main_Table/10core/K:10/Movies_and_TV/ \
#     --model_idx Mean\
#     --contrast_type Item-Level \
#     --warm_up_epoches 0\
#     --rec_weight 1 \
#     --temperature 1 \
#     --num_intent_clusters 10\
#     --intent_cf_weight 1.2\
#     --cf_weight 0 \
#     --cluster_value 0.3\

# # scripts/Beauty/0.sh
