from config_param import config

import pandas as pd
import numpy as np

df = pd.read_csv(config.processed_data_path, usecols=['ID','Date','bin_pink','polarity','subjectivity','intensity'])

print('\nSplitting Data...')
w_pink = df[df.bin_pink == 1]
wo_pink = df[df.bin_pink == 0]

baseline = wo_pink.groupby(['ID'], as_index=False)['polarity'].mean().rename(columns = {'polarity': 'base_polarity'})
w_pink = pd.merge(w_pink, baseline, how = 'left', on = 'ID')

baseline = wo_pink.groupby(['ID'], as_index=False)['subjectivity'].mean().rename(columns = {'subjectivity': 'base_subjectivity'})
w_pink = pd.merge(w_pink, baseline, how = 'left', on = 'ID')

##For this analysis, we must drop bloggers with only 1 post
print('\nShape before dropping Na values: {}'.format(w_pink.shape))
w_pink = w_pink[np.isfinite(w_pink.base_polarity)]
print('\nShape after dropping Na values: {}'.format(w_pink.shape))

print('\nExporting final dataset')
w_pink.to_csv(config.sentiments_with_baseline)