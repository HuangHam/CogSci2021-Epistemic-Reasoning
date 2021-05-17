import pandas as pd
from src.agents import *
from src.procedures import Simulator
'''
Run this file to simulate using fitted parameters
Used nSim=100 in the paper but github caps file size at 100mb
So here report nSim=20
'''
data = pd.read_csv("data.csv")

agents = [NoisyDEL,SIWEB,SUWNB,SUWEB]

for agent in agents:
    sim = Simulator(agent(data), nSim=20) 
    sim.validation_simulation()

