

python main.py \
    --model_name UPTRec \
    --data_name Sports_and_Outdoors  \
    --data_dir data/ \
    --context encoder \
    --seq_representation_type concatenate \
    --attention_type None \
    --de_noise \
    --batch_size 512 \
    --epochs 2000 \
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 30 \
    --embedding \
    --output_dir Main_Table/Sports_and_Outdoors/SASRec \
    --model_idx SASRec\
    --contrast_type None \


python main.py \
    --model_name UPTRec \
    --data_name Toys_and_Games  \
    --data_dir data/ \
    --context encoder \
    --seq_representation_type concatenate \
    --attention_type None \
    --de_noise \
    --batch_size 512 \
    --epochs 2000 \
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 30 \
    --embedding \
    --output_dir Main_Table/Toys_and_Games/SASRec \
    --model_idx SASRec\
    --contrast_type None \

python main.py \
    --model_name UPTRec \
    --data_name Yelp  \
    --data_dir data/ \
    --context encoder \
    --seq_representation_type concatenate \
    --attention_type None \
    --de_noise \
    --batch_size 512 \
    --epochs 2000 \
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 30 \
    --embedding \
    --output_dir Main_Table/Yelp/SASRec \
    --model_idx SASRec\
    --contrast_type None \

python main.py \
    --model_name UPTRec \
    --data_name ml-1m  \
    --data_dir data/ \
    --context encoder \
    --seq_representation_type concatenate \
    --attention_type None \
    --de_noise \
    --batch_size 512 \
    --epochs 2000 \
    --gpu_id 1 \
    --visualization_epoch 20 \
    --patience 30 \
    --embedding \
    --max_seq_length 200 \
    --output_dir Main_Table/ml-1m/SASRec \
    --model_idx SASRec\
    --contrast_type None \

# scripts/Beauty/Contrastive_Learning_ablation.sh