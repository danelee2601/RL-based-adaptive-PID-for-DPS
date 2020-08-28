

class ThrusterData(object):
    # This class is used to store and share the control parameters
    # and the resulting thruster load:
    def __init__(self):
        # These constants are defined in the vessel's external function parameters
        # in the OrcaFlex model:
        self.TargetX = 0.0
        self.TargetY = 0.0
        self.TargetHeading = 0.0

        # gains from Orca file
        self.Kp_x = 0.
        self.Kd_x = 0.
        self.Ki_x = 0.
        self.Kp_y = 0.
        self.Kd_y = 0.
        self.Ki_y = 0.
        self.mp = 0.
        self.md = 0.
        self.mi = 0.

        # The thruster load:
        self.ForceX = 0.0
        self.ForceY = 0.0
        self.MomentZ = 0.0

        # dpt (from dp_force.cpp)
        self.dpt_zero = 0.0
        self.dpt_one = 0.0
        self.dpt_two = 0.0
        self.dpt_three = 0.0
        self.dpt_four = 0.0
        self.dpt_five = 0.0
        self.dpt_six = 0.0
        self.dpt_seven = 0.0
        self.dpt_eight = 0.0

        # w (from dp_force.cpp)
        # No.1 Thruster
        self.w_zero_zero = 0.0
        self.w_zero_one = 0.0
        self.w_zero_two = 0.0
        self.w_zero_three = 0.0
        self.w_zero_four = 0.0
        self.w_zero_five = 0.0
        self.w_zero_six = 0.0

        # No.2 Thruster
        self.w_one_zero = 0.0
        self.w_one_one = 0.0
        self.w_one_two = 0.0
        self.w_one_three = 0.0
        self.w_one_four = 0.0
        self.w_one_five = 0.0
        self.w_one_six = 0.0

        # No.3 Thruster
        self.w_two_zero = 0.0
        self.w_two_one = 0.0
        self.w_two_two = 0.0
        self.w_two_three = 0.0
        self.w_two_four = 0.0
        self.w_two_five = 0.0
        self.w_two_six = 0.0

        # No.4 Thruster
        self.w_three_zero = 0.0
        self.w_three_one = 0.0
        self.w_three_two = 0.0
        self.w_three_three = 0.0
        self.w_three_four = 0.0
        self.w_three_five = 0.0
        self.w_three_six = 0.0

        # No.5 Thruster
        self.w_four_zero = 0.0
        self.w_four_one = 0.0
        self.w_four_two = 0.0
        self.w_four_three = 0.0
        self.w_four_four = 0.0
        self.w_four_five = 0.0
        self.w_four_six = 0.0

        # No.6 Thruster
        self.w_five_zero = 0.0
        self.w_five_one = 0.0
        self.w_five_two = 0.0
        self.w_five_three = 0.0
        self.w_five_four = 0.0
        self.w_five_five = 0.0
        self.w_five_six = 0.0

        # No.7 Thruster
        self.w_six_zero = 0.0
        self.w_six_one = 0.0
        self.w_six_two = 0.0
        self.w_six_three = 0.0
        self.w_six_four = 0.0
        self.w_six_five = 0.0
        self.w_six_six = 0.0

        # No.8 Thruster
        self.w_seven_zero = 0.0
        self.w_seven_one = 0.0
        self.w_seven_two = 0.0
        self.w_seven_three = 0.0
        self.w_seven_four = 0.0
        self.w_seven_five = 0.0
        self.w_seven_six = 0.0
