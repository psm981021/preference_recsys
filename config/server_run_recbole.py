
from recbole.quick_start import run_recbole

# run_recbole(model='BERT4Rec', dataset='Amazon_Beauty', config_file_list=['BERT4Rec.yaml'])
# run_recbole(model='SASRec', dataset='Amazon_Beauty', config_file_list=['SASRec.yaml'])


datasets = ['Amazon_Clothing_Shoes_and_Jewelry','Amazon_Sports_and_Outdoors','Amazon_Tools_and_Home_Improvement','Amazon_Video_Games']



### BERT4Rec ### 
# for dataset in datasets:
#     run_recbole(model='BERT4Rec', dataset=dataset, config_file_list=['BERT4Rec.yaml'])


### Caser ###

run_recbole(model='Caser', dataset='Amazon_Clothing_Shoes_and_Jewelry', config_file_list=['Caser.yaml'])
run_recbole(model='Caser', dataset='Amazon_Sports_and_Outdoors', config_file_list=['Caser.yaml'])
run_recbole(model='Caser', dataset='Amazon_Tools_and_Home_Improvement', config_file_list=['Caser.yaml'])
run_recbole(model='Caser', dataset='Amazon_Video_Games', config_file_list=['Caser.yaml'])


### ADMMSLIM ### 

run_recbole(model='ADMMSLIM', dataset='Amazon_Clothing_Shoes_and_Jewelry', config_file_list=['ADMMSLIM.yaml'])
run_recbole(model='ADMMSLIM', dataset='Amazon_Sports_and_Outdoors', config_file_list=['ADMMSLIM.yaml'])
run_recbole(model='ADMMSLIM', dataset='Amazon_Tools_and_Home_Improvement', config_file_list=['ADMMSLIM.yaml'])
run_recbole(model='ADMMSLIM', dataset='Amazon_Video_Games', config_file_list=['ADMMSLIM.yaml'])



# python server_run_recbole.py