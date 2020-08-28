import os, sys, re
sys.path.append(re.search(r'.*?\\DP', os.path.dirname(__file__)).group(0))  # register the root dir

from DP_DDPG.thruster_data import ThrusterData as DDPGThrusterData


class ThrusterData(DDPGThrusterData):

    def __init__(self):
        pass
