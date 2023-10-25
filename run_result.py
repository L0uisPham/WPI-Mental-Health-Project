import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from utils import load_results 
from train_read_models import *


total_df = read_all_unstratified_models(
    models=['RFC'],
    study_dir_name='April_18',
    version='summary',
)





