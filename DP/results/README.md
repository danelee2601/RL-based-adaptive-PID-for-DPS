# Description
---
### train/reward_func_case1
File/Folder | description
----------- | -----------
models | contains a trained model of the proposed controller
gain_hist.csv | contains time histories of the adaptive PID gains and the update-gates for the integral of the errors during training
ship_pos.csv | contains time histories of ship motions, velocities, and the integral of the errors w.r.t surge, sway, and yaw directions during training
thrust_hist.csv | contains time histories of thrust [N] from the six azimuth thrusters during training
training_hist.csv | contains time histories of the reward and Q function, and loss time histories of the actor and critic during training

### test/reward_func_case1
File/Folder | description
----------- | -----------
models | empty
gain_hist.csv | same as above except that it is subject to testing
ship_pos.csv | same as above except that it is subject to testing
thrust_hist.csv | same as above except that it is subject to testing
training_hist.csv | empty (all values are zero)

### ZN/temp
File/Folder | description
----------- | -----------
gain_hist.csv | same as above except that it is obtained by a PID controller with the fixed base gain
pred_history.csv | irrelevant (please ignore this)
ship_pos.csv |same as above except that it is obtained by a PID controller with the fixed base gain
thrust_hist.csv | same as above except that it is obtained by a PID controller with the fixed base gain
training_hist.csv | empty