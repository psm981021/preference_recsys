# for augment_type in mask
# do
#     for gamma in 0.7 
#     do
#         python main.py \
#             --model_name="UPTRec" \
#             --data_name="Beauty"  \
#             --output_dir="output_custom/Beauty" \
#             --contrast_type="IntentCL" \
#             --seq_representation_type="concatenate" \
#             --attention_type="Cluster" \
#             --model_idx="UPTRec_Clustered_Attention_IntentCL_${augment_type}_${gamma}" \
#             --augment_type=$augment_type \
#             --gamma=$gamma \
#             --num_intent_clusters=16 --gpu_id=0 --epochs=1000 --patience=500 --warm_up_epochs=40 --de_noise

#     done
# done


python main.py \
    --model_name="UPTRec" \
    --data_name="Beauty"  \
    --augment_type="mask" \
    --gamma=0.7 \
    --n_views=3 \
    --temperature=0.5 \
    --output_dir="output_custom/Beauty/Cluster_Attention_IntentCL" \
    --contrast_type="IntentCL" \
    --seq_representation_type="concatenate" \
    --attention_type="Cluster" \
    --model_idx="UPTRec_Clustered_Attention_IntentCL_${augment_type}_${gamma}" \
    --num_intent_clusters=16 --gpu_id=1 --epochs=1000 --patience=500 --warm_up_epoches=40 #--do_eval --embedding
