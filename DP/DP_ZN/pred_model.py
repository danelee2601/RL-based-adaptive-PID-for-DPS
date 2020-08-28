import os
from collections import deque

import numpy as np
import pandas as pd

import tensorflow.keras.backend as K
from tensorflow import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class PredModel(object):

    def __init__(self, dps_settings):
        self.dps_settings = dps_settings

        # set attributes - model
        self.n_features = 15
        self.n_output = 3
        self.dropout_rate = 0.4
        self.n_starting_epochs = 30
        self.validation_split = 0.2
        self.n_epochs = 1  # per training
        self.verbose_period = 100  # [step]

        # params for training
        self.use_replay_buffer = True  # recommended: True  # False: train with the latest data only
        self.starting_training = False  # (default) True, initial training for 'n_starting_epochs'
        self.scale_obs = True

        # set attributes - model prediction
        self.use_pred_2nd_derivative_err = False  # recommended: False
        self.use_pred_integral_err = False  # recommended: False
        self.past_length = 50  # [step]
        self.sampling_rate = 5  # [step]
        self.pred_length = 10  # [s]
        self.model_type = 'lstm'  # dense / lstm / gru
        self.hl_size = 64  # hidden layer size
        self.n_hidden_layers = 1

        # params for amplifier
        self.use_amplifier = False  # recommended: False
        self.amp_weight = 1.8
        self.amp_cutoff_y = 20

        # set attributes - pred model memory
        self.starting_memory_size = 200e1 if self.starting_training else 1  # (default) 200e1
        self.memory_size = 3600e1
        self.batch_size = 256
        self.online_batch_size = 1

        # params for saving data
        self.f_training_hist = open(os.path.join(self.dps_settings.dirfullname, 'training_history.csv'), 'a')
        self.f_training_hist.write('epoch,loss,r2_score,val_loss,val_r2_score\n')
        self.f_pred_hist = open(os.path.join(self.dps_settings.dirfullname, 'pred_history.csv'), 'a')
        self.f_pred_hist.write('timestep,x,y,r3\n')

        # interactive attributes
        self.model = ''
        self.n_training = 0
        self.pred_memory = {name: deque([0., 0., 0.], maxlen=3) for name in ['x', 'y', 'r3']}
        self.active_status = False

    def r2_keras(self, y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - SS_res / (SS_tot + K.epsilon())

    def update_model(self):
        with tf.name_scope('placeholders'):
            self.ph_X = tf.placeholder(dtype=tf.float32,
                                       shape=(None, self.past_length // self.sampling_rate, self.n_features))
            self.ph_Y = tf.placeholder(dtype=tf.float32, shape=(None, self.n_output))

        with tf.name_scope('model'):
            if self.model_type == 'dense':
                nn = tf.keras.layers.Flatten()(self.ph_X)
                for i in range(self.n_hidden_layers):
                    nn = tf.keras.layers.Dense(self.hl_size, activation='relu', kernel_initializer='he_normal')(nn)
                    nn = tf.keras.layers.Dropout(self.dropout_rate)(nn)
                self.model = tf.keras.layers.Dense(self.n_output)(nn)
            elif self.model_type == 'lstm':
                nn = tf.keras.layers.LSTM(self.hl_size,
                                          return_sequences=False if self.n_hidden_layers == 1 else True,
                                          dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate)(self.ph_X)
                for i in range(self.n_hidden_layers - 1):
                    nn = tf.keras.layers.LSTM(self.hl_size,
                                              return_sequences=True if i != (self.n_hidden_layers - 2) else False,
                                              dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate)(nn)
                self.model = tf.keras.layers.Dense(self.n_output)(nn)
            elif self.model_type == 'gru':
                nn = tf.keras.layers.GRU(self.hl_size,
                                         return_sequences=False if self.n_hidden_layers == 1 else True,
                                         dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate)(self.ph_X)
                for i in range(self.n_hidden_layers - 1):
                    nn = tf.keras.layers.GRU(self.hl_size,
                                             return_sequences=True if i != (self.n_hidden_layers - 2) else False,
                                             dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate)(nn)
                self.model = tf.keras.layers.Dense(self.n_output)(nn)

        with tf.name_scope('compile'):
            self.cost = tf.reduce_mean((self.model - self.ph_Y) ** 2)
            total_error = tf.reduce_sum(tf.square(tf.subtract(self.ph_Y, tf.reduce_mean(self.ph_Y))))
            unexplained_error = tf.reduce_sum(tf.square(tf.subtract(self.ph_Y, self.model)))
            self.R_squared = tf.subtract(1., tf.div(unexplained_error, total_error))
            self.op = tf.train.AdamOptimizer(1e-3).minimize(self.cost)

        with tf.name_scope('train_init'):
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def fit(self, time_series_gen, time_series_gen_val, epochs, verbose):
        print_fit_progress = "epoch:{}, loss:{:0.3f}, r2_score:{:0.3f}"

        for epoch in range(epochs):
            # train
            loss_hist, r2_score_hist = [], []
            for x_batch, y_batch in time_series_gen:
                _, loss, r2_score = self.sess.run([self.op, self.cost, self.R_squared],
                                                  feed_dict={self.ph_X: x_batch, self.ph_Y: y_batch})
                loss_hist.append(loss)
                r2_score_hist.append(r2_score)

            mean_loss, mean_r2_score = np.mean(loss_hist), np.mean(r2_score_hist)
            print(print_fit_progress.format(epoch, mean_loss, mean_r2_score)) if verbose else None

            # val
            if time_series_gen_val:
                loss_hist, r2_score_hist = [], []
                for x_val_batch, y_val_batch in time_series_gen_val:
                    # train
                    loss, r2_score = self.sess.run([self.cost, self.R_squared],
                                                   feed_dict={self.ph_X: x_val_batch, self.ph_Y: y_val_batch})
                    loss_hist.append(loss)
                    r2_score_hist.append(r2_score)

                val_mean_loss, val_mean_r2_score = np.mean(loss_hist), np.mean(r2_score_hist)
                print(print_fit_progress.format(epoch, val_mean_loss, val_mean_r2_score) + '- val',
                      end='\n\n') if verbose else None

            # save data
            self.f_training_hist.write(
                f"{epoch},{mean_loss},{mean_r2_score},{val_mean_loss if time_series_gen_val else None},{val_mean_r2_score if time_series_gen_val else None}\n")

    def to_csv(self, arr_x, fname_x, columns):
        pd.DataFrame(arr_x, columns=columns).to_csv(os.path.join(self.dps_settings.dirfullname, f'{fname_x}.csv'),
                                                    index=None)

    def get_batchXY_gen(self, batchX, batchY):
        yield batchX, batchY

    def save_y_hat_y_true(self, sc_memoryX, sc_memoryY, pred_model_memory, val_data=False):
        """
        save y_hat, y_true for comparision
        """
        time_series_gen = keras.preprocessing.sequence.TimeseriesGenerator(
            data=sc_memoryX, targets=sc_memoryY,
            length=self.past_length, sampling_rate=self.sampling_rate, batch_size=10000000000)

        if time_series_gen[1][0].size != 0:  # check if all the data got into the first iter in time_series_gen
            raise ValueError

        scX, scY = time_series_gen[0][0], time_series_gen[0][1]
        sc_y_hat = self.sess.run([self.model], feed_dict={self.ph_X: scX})[0]
        y_hat = sc_y_hat * pred_model_memory.sigma_Y + pred_model_memory.mu_Y
        y_true = scY * pred_model_memory.sigma_Y + pred_model_memory.mu_Y

        # export
        self.to_csv(y_hat, 'y_hat' if not val_data else 'val_y_hat', ['x', 'y', 'r3'])
        self.to_csv(y_true, 'y_true' if not val_data else 'val_y_true', ['x', 'y', 'r3'])

    def train_model(self, pred_model_memory):
        if (pred_model_memory.len_memory >= self.batch_size) and (pred_model_memory.len_memory >= self.starting_memory_size):

            mu_X, sigma_X = pred_model_memory.mu_X, pred_model_memory.sigma_X
            mu_Y, sigma_Y = pred_model_memory.mu_Y, pred_model_memory.sigma_Y

            if (self.n_training == 0) and (self.starting_training == True):

                # data scaling (normalization)
                if self.scale_obs:
                    sc_memoryX = (np.array(pred_model_memory.memoryX) - mu_X) / sigma_X
                    sc_memoryY = (np.array(pred_model_memory.memoryY) - mu_Y) / sigma_Y
                else:
                    sc_memoryX = np.array(pred_model_memory.memoryX)
                    sc_memoryY = np.array(pred_model_memory.memoryY)

                # prepare the training / validation datasets
                cutoff_idx = int(sc_memoryX.shape[0] * (1 - self.validation_split))
                sc_memoryX, sc_memoryX_val = sc_memoryX[:cutoff_idx], sc_memoryX[cutoff_idx:]
                sc_memoryY, sc_memoryY_val = sc_memoryY[:cutoff_idx], sc_memoryY[cutoff_idx:]
                time_series_gen = keras.preprocessing.sequence.TimeseriesGenerator(
                    data=sc_memoryX, targets=sc_memoryY,
                    length=self.past_length, sampling_rate=self.sampling_rate, batch_size=self.batch_size)
                time_series_gen_val = keras.preprocessing.sequence.TimeseriesGenerator(
                    data=sc_memoryX_val, targets=sc_memoryY_val,
                    length=self.past_length, sampling_rate=self.sampling_rate, batch_size=self.batch_size)
                self.fit(time_series_gen, time_series_gen_val, self.n_starting_epochs, verbose=1)

                # save y_hat, y_true to validate the prediction accuracy
                self.save_y_hat_y_true(sc_memoryX, sc_memoryY, pred_model_memory)
                self.save_y_hat_y_true(sc_memoryX_val, sc_memoryY_val, pred_model_memory, val_data=True)
            else:
                mini_batchX, mini_batchY = pred_model_memory.fetch_mini_batch(mu_X, sigma_X, mu_Y, sigma_Y)
                batchXY_gen = self.get_batchXY_gen(mini_batchX, mini_batchY)
                self.fit(batchXY_gen, None, self.n_epochs,
                         verbose=1 if self.n_training % self.verbose_period == 0 else 0)
            self.n_training += 1

    def predict(self, x):
        pred = self.sess.run([self.model], feed_dict={self.ph_X: x})[0][0]
        return pred

    def is_towards_target(self, current_global_pos, current_global_vel):
        """
        current_global_pos: [x, y, r3]
        current_global_vel: [vel_x, vel_y, vel_r3]

        The global pos, vel are converted to local ones and
        if the vessel is going towards a target pos, True is returned else False.
        It works effectively when either kalman filter is applied or 1st wave order effect is ignored.
        """
        # convert global coords/vel to local coords/vel
        heading_angle = current_global_pos[2]
        R = np.array([[np.cos(heading_angle), -np.sin(heading_angle), 0],
                      [np.sin(heading_angle), np.cos(heading_angle), 0],
                      [0, 0, 1]])

        current_global_pos = np.array(current_global_pos).reshape(-1, 1)
        current_global_vel = np.array(current_global_vel).reshape(-1, 1)
        current_local_pos = np.dot(R, current_global_pos).ravel()
        current_local_vel = np.dot(R, current_global_vel).ravel()

        answer = {}
        for clp, clv, dir in zip(current_local_pos, current_local_vel, ['x', 'y', 'r3']):
            if clp > 0 and clv < 0:
                answer[dir] = True
            elif clp < 0 and clv > 0:
                answer[dir] = True
            else:
                answer[dir] = False
        return answer

    def amplifier(self, x):
        """
        x: input
        y: ouput
        """
        if x <= 1.0:  # liner where x <= 1.0
            y = x
        elif (1.0 < x) and (x ** self.amp_weight <= self.amp_cutoff_y):  # exponential section
            y = x ** self.amp_weight
        else:
            conn_intercept = self.amp_cutoff_y - self.amp_cutoff_y ** (1 / self.amp_weight)
            y = x + conn_intercept
        return y


class PredModelMemory(object):

    def __init__(self, dps_settings, pred_model):
        # given attributes - class instance
        self.dps_settings = dps_settings
        self.pred_model = pred_model

        # interactive attributes
        self.memoryX = []
        self.memoryY = []
        self.len_memory = 0
        self.mu_X = np.zeros(self.pred_model.n_features)
        self.var_X = np.zeros(self.pred_model.n_features)
        self.sigma_X = np.zeros(self.pred_model.n_features)  # std
        self.mu_Y = np.zeros(self.pred_model.n_output)
        self.var_Y = np.zeros(self.pred_model.n_output)
        self.sigma_Y = np.zeros(self.pred_model.n_output)

        self.pos_hist = {name: [] for name in ['X', 'Y', 'R3']}
        self.vel_hist = {name: [] for name in ['velX', 'velY', 'velR3']}
        self.acc_hist = {name: [] for name in ['accX', 'accY', 'accR3']}
        self.thrust_hist = {name: [] for name in ['Tx', 'Ty', 'Mz']}
        self.wind_hist = {name: [] for name in ['wfX', 'wfY', 'wfR3']}

    def store_hist(self, x, y, r3, vel_x, vel_y, vel_r3, acc_x, acc_y, acc_r3, t_x, t_y, m_z, wf_x, wf_y, wf_r3):
        """
        store history in dictionaries
        """
        self.pos_hist['X'].append(x)
        self.pos_hist['Y'].append(y)
        self.pos_hist['R3'].append(r3)
        self.vel_hist['velX'].append(vel_x)
        self.vel_hist['velY'].append(vel_y)
        self.vel_hist['velR3'].append(vel_r3)
        self.acc_hist['accX'].append(acc_x)
        self.acc_hist['accY'].append(acc_y)
        self.acc_hist['accR3'].append(acc_r3)
        self.thrust_hist['Tx'].append(t_x)
        self.thrust_hist['Ty'].append(t_y)
        self.thrust_hist['Mz'].append(m_z)
        self.wind_hist['wfX'].append(wf_x)
        self.wind_hist['wfY'].append(wf_y)
        self.wind_hist['wfR3'].append(wf_r3)

        # manage hist size
        maxlen_memory = self.pred_model.pred_length * int(1 / self.dps_settings.SimulationTimeStep) + 10
        if len(self.pos_hist['X']) > maxlen_memory:
            self.pos_hist['X'] = self.pos_hist['X'][1:]
            self.pos_hist['Y'] = self.pos_hist['Y'][1:]
            self.pos_hist['R3'] = self.pos_hist['R3'][1:]
            self.vel_hist['velX'] = self.vel_hist['velX'][1:]
            self.vel_hist['velY'] = self.vel_hist['velY'][1:]
            self.vel_hist['velR3'] = self.vel_hist['velR3'][1:]
            self.acc_hist['accX'] = self.acc_hist['accX'][1:]
            self.acc_hist['accY'] = self.acc_hist['accY'][1:]
            self.acc_hist['accR3'] = self.acc_hist['accR3'][1:]
            self.thrust_hist['Tx'] = self.thrust_hist['Tx'][1:]
            self.thrust_hist['Ty'] = self.thrust_hist['Ty'][1:]
            self.thrust_hist['Mz'] = self.thrust_hist['Mz'][1:]
            self.wind_hist['wfX'] = self.wind_hist['wfX'][1:]
            self.wind_hist['wfY'] = self.wind_hist['wfY'][1:]
            self.wind_hist['wfR3'] = self.wind_hist['wfR3'][1:]

    def update_mu_sigma(self, X, Y, epsilon=1e-10):
        """
        update mu, var, sigma(std)
        by calculating iterative mean, std eqs
        reference url: https://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
        """
        X, Y = np.array(X), np.array(Y)

        self.mu_X = ((self.len_memory - 1) / self.len_memory) * self.mu_X + (1 / self.len_memory) * X
        self.var_X = ((self.len_memory - 2) / (self.len_memory - 1 + epsilon)) * self.var_X + \
                     (1 / self.len_memory) * (X - self.mu_X) ** 2
        self.sigma_X = np.sqrt(self.var_X)

        self.mu_Y = ((self.len_memory - 1) / self.len_memory) * self.mu_Y + (1 / self.len_memory) * Y
        self.var_Y = ((self.len_memory - 2) / (self.len_memory - 1 + epsilon)) * self.var_Y + \
                     (1 / self.len_memory) * (Y - self.mu_Y) ** 2
        self.sigma_Y = np.sqrt(self.var_Y)

    def store_memory(self, X, Y):
        """
        store a training set of (X, Y) in the replay buffer
        """
        self.memoryX.append(X)
        self.memoryY.append(Y)
        self.len_memory += 1
        self.update_mu_sigma(X, Y)

        # manage memory size
        if self.len_memory > self.pred_model.memory_size:
            self.memoryX = self.memoryX[1:]
            self.memoryY = self.memoryY[1:]
            self.len_memory -= 1

    def fetch_mini_batch(self, mu_X, sigma_X, mu_Y, sigma_Y):
        mini_batchX, mini_batchY = [], []
        while len(mini_batchX) <= self.pred_model.online_batch_size:

            pick_idx = np.random.randint(0, self.len_memory)
            while pick_idx + self.pred_model.past_length > self.len_memory - 1:
                pick_idx = np.random.randint(0, self.len_memory)

            if not self.pred_model.use_replay_buffer:
                pick_idx = (self.len_memory - 1) - self.pred_model.past_length

            mbX = self.memoryX[pick_idx: pick_idx + self.pred_model.past_length: self.pred_model.sampling_rate]
            mbY = self.memoryY[pick_idx + self.pred_model.past_length]

            # normalize
            mbX = (mbX - mu_X) / sigma_X
            mbY = (mbY - mu_Y) / sigma_Y

            mini_batchX.append(mbX)
            mini_batchY.append(mbY)

        mini_batchX, mini_batchY = np.array(mini_batchX), np.array(mini_batchY)
        return mini_batchX, mini_batchY
