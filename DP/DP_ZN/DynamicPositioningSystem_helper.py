import os, sys, re
sys.path.append(re.search(r'.*?\\DP', os.path.dirname(__file__)).group(0))  # register the root dir

import numpy as np
from DP_DDPG.DynamicPositioningSystem_helper import DPSHelper as DDPGDPSHelper


class DPSHelper(DDPGDPSHelper):

    def __init__(self, dps_settings):
        super().__init__(dps_settings)

        self.dps_settings = dps_settings


    def print_current_ship_DPstate(self, timestep):
        print("\n\n==================================================================")
        print("timestep: {:0.1f}".format(timestep))
        print("accumulated_local_err['X']: {:0.2f}, accumulated_local_err['Y']: {:0.2f}, accumulated_local_err['R3']: {:0.2f}".format( *list(self.dps_settings.accumulated_local_err.values()) ))

