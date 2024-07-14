
from recbole.quick_start import run_recbole

# run_recbole(model='BERT4Rec', dataset='Amazon_Beauty', config_file_list=['BERT4Rec.yaml'])
# run_recbole(model='SASRec', dataset='Amazon_Beauty', config_file_list=['SASRec.yaml'])





run_recbole(model='Caser', dataset='Amazon_Beauty', config_file_list=['Caser.yaml'])
# run_recbole(model='S3Rec', dataset='Amazon_Beauty', config_file_list=['S3Rec.yaml'])

# run_recbole(model='S3Rec', dataset='Amazon_Beauty',config_dict=config_dict, saved=False)

# python server_run_recbole.py