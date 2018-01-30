"""
__file__
	run_all.py
__description___

	This file generates all the features in one shot.
__author__
	Dan EMery < emeryde@appstate.edu >
"""

import os

#################
## Preprocesss ##
#################
#### preprocess data
cmd = "py -3.6 preprocess.py"
os.system(cmd)


#### sentiment feats
cmd = "py -3.6 sentiment_baseline.py"
os.system(cmd)

#### comparison feats
cmd = "py -3.6 sentiment_comparison.py"
os.system(cmd)