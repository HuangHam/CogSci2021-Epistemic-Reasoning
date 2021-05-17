import pandas as pd
import numpy as np
import time
from src.procedures import Fitter
from src.agents import *
'''
Run this file to fit models 
Note: this is computationally intensive. Can take up to days to fit
We used more than 100 cores to fit parameters, which took several hours (for the SUWEB model)
'''
data = pd.read_csv("data.csv")

agents = [NoisyDEL,SIWEB,SUWNB,SUWEB]
			
start_time = time.time() # register the start time
for agent in agents:
    fit = Fitter(agent(data))
    nSub = np.arange(1, fit.num_subj+1) # fit all subjects
    fit.parallel_fit(nSub, nCore=1) # parallel fitting. Computationally intensive

print("--- %s seconds ---" % (time.time() - start_time))





