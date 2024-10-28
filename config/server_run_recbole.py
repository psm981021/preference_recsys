
from recbole.quick_start import run_recbole

# run_recbole(model='BERT4Rec', dataset='Amazon_Beauty', config_file_list=['BERT4Rec.yaml'])
# run_recbole(model='SASRec', dataset='Amazon_Beauty', config_file_list=['SASRec.yaml'])


datasets = ['Amazon_Clothing_Shoes_and_Jewelry','Amazon_Sports_and_Outdoors','Amazon_Tools_and_Home_Improvement','Amazon_Video_Games']



### BERT4Rec ### 

# run_recbole(model='BERT4Rec', dataset='Amazon_Clothing_Shoes_and_Jewelry', config_file_list=['BERT4Rec.yaml'])
# run_recbole(model='BERT4Rec', dataset='Amazon_Sports_and_Outdoors', config_file_list=['BERT4Rec.yaml'])
# run_recbole(model='BERT4Rec', dataset='Amazon_Tools_and_Home_Improvement', config_file_list=['BERT4Rec.yaml'])
# run_recbole(model='BERT4Rec', dataset='Amazon_Pet_Supplies', config_file_list=['BERT4Rec.yaml'])
# run_recbole(model='BERT4Rec', dataset='Ml-1m', config_file_list=['BERT4Rec.yaml'])


### Caser ###

# run_recbole(model='Caser', dataset='Amazon_Clothing_Shoes_and_Jewelry', config_file_list=['Caser.yaml'])
# run_recbole(model='Caser', dataset='Amazon_Sports_and_Outdoors', config_file_list=['Caser.yaml'])
# run_recbole(model='Caser', dataset='Amazon_Tools_and_Home_Improvement', config_file_list=['Caser.yaml'])
# run_recbole(model='Caser', dataset='Amazon_Video_Games', config_file_list=['Caser.yaml'])
# run_recbole(model='Caser', dataset='Amazon_Pet_Supplies', config_file_list=['Caser.yaml'])
run_recbole(model='Caser', dataset='Ml-1m', config_file_list=['Caser.yaml'])



### GRU4Rec ### 

# run_recbole(model='GRU4Rec', dataset='Amazon_Clothing_Shoes_and_Jewelry', config_file_list=['GRU4Rec.yaml'])
# run_recbole(model='GRU4Rec', dataset='Amazon_Sports_and_Outdoors', config_file_list=['GRU4Rec.yaml'])
# run_recbole(model='GRU4Rec', dataset='Amazon_Tools_and_Home_Improvement', config_file_list=['GRU4Rec.yaml'])
# run_recbole(model='GRU4Rec', dataset='Amazon_Video_Games', config_file_list=['GRU4Rec.yaml'])
# run_recbole(model='GRU4Rec', dataset='Amazon_Pet_Supplies', config_file_list=['GRU4Rec.yaml'])
run_recbole(model='GRU4Rec', dataset='Ml-1m', config_file_list=['GRU4Rec.yaml'])

### FEARec ###
# run_recbole(model='FEARec', dataset='Amazon_Sports_and_Outdoors', config_file_list=['FEARec.yaml'])
# run_recbole(model='FEARec', dataset='Amazon_Clothing_Shoes_and_Jewelry', config_file_list=['FEARec.yaml'])
# run_recbole(model='FEARec', dataset='Amazon_Tools_and_Home_Improvement', config_file_list=['FEARec.yaml'])
# run_recbole(model='FEARec', dataset='Amazon_Video_Games', config_file_list=['FEARec.yaml'])


# python server_run_recbole.py