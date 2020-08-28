import os, sys, re
sys.path.append(re.search(r'.*?\\DP', os.path.dirname(__file__)).group(0))  # register the root dir

from DP_DDPG.proc_input import ProcInput as DDPGProcInput
from DP_DDPG.proc_input import SimplifyVel as DDPGSimplifyVel


class ProcInput(DDPGProcInput):
    def __init__(self, dps_settings):
        super().__init__(dps_settings)


class SimplifyVel(DDPGSimplifyVel):
    def __init__(self, sampling_period):
        super().__init__(sampling_period)
