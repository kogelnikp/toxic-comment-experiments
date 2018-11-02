import pandas as pd
import os
import sys

script_dir = os.path.dirname(__file__)
data_test_path = os.path.join(script_dir, '../data/kaggle/test.csv')
data_labels_path = os.path.join(script_dir, '../data/kaggle/test_labels.csv')
data_result_path = os.path.join(script_dir, '../data/kaggle/test_complete.csv')

if not os.path.isfile(data_test_path):
    print('../data/kaggle/test.csv not found')
    sys.exit()

if not os.path.isfile(data_labels_path):
    print('../data/kaggle/test_labels.csv not found')
    sys.exit()

print('Read test.csv and test_labels.csv...')
data_test = pd.read_csv(data_test_path, index_col=0)
data_labels = pd.read_csv(data_labels_path, index_col=0)

print('Combine data from both files...')
data_combined = pd.concat([data_test, data_labels], axis=1)

print('Only keep rows where toxic != -1...')
data_filtered = data_combined[data_combined['toxic'] != -1]

print('Save resulting file as test_complete.csv...')
data_filtered.to_csv(data_result_path)

print('Done!')