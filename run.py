from recbole.config import Config
from recbole.quick_start import run_recbole
# parameter_dict={
#     'data_path':'/Users/sb/Desktop/project/preference_rec/dataset',
#     'train_neg_sample_args':None,
# }
run_recbole(model='SASRec', dataset='Amazon_Sports_and_Outdoors', config_file_list=['sasrec.yaml'])
