from config_param import config

import pandas as pd
import numpy as np

df = pd.read_csv(config.sentiments_with_baseline)
def get_polarity_comparison(x):
    score = x['polarity']
    base = x['base_polarity']
    if (score > base):
        return 1
    else:
        return 0

def get_subjectivity_comparison(x):
    score = x['subjectivity']
    base = x['base_subjectivity']
    if (score > base):
        return 1
    else:
        return 0

df['comp_polarity'] = list(df.apply(get_polarity_comparison, axis = 1))
df['comp_subjectivity'] = list(df.apply(get_subjectivity_comparison, axis = 1))

print(np.mean(df.comp_polarity))
##We can see that 25.6% or more positive when talking about pink

print(np.mean(df.comp_subjectivity))
##We can see that 57.8% of sentences are more subjective when describing pink

df.to_csv('output/sentiments_final.csv', index = False, header = True)