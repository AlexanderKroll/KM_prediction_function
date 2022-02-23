import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, add
from tensorflow.keras.losses import MSE
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.activations import relu
from tensorflow.keras import backend as K
import os
from os.path import join

import warnings
warnings.filterwarnings("ignore")

CURRENT_DIR = os.getcwd()
metabolite_dir = join(CURRENT_DIR, "..", "data", "temp_met")

def calculate_gnn_representations(df_met):
	N_max = np.max(df_met["number_atoms"].loc[df_met["successfull"]]) + 1
	GNN_representation_fct = loading_GNN(N_max)

	df_valid_met = df_met.loc[df_met["successfull"]]
	df_invalid_met = df_met.loc[~df_met["successfull"]]

	df_valid_met.reset_index(inplace = True, drop = True)
	df_valid_met = get_metabolite_representations(df = df_valid_met, get_fingerprint_fct = GNN_representation_fct)
	df_invalid_met["GNN rep"] = ""
	df_met = pd.concat([df_valid_met, df_invalid_met], ignore_index = True)
	return(df_met)

def loading_GNN(N_max):
	batch_size, D, learning_rate, epochs, l2_reg_fc, l2_reg_conv, rho = 64, 50, 0.05, 50, 1, 0.01, 0.95
	model = DMPNN(l2_reg_conv = l2_reg_conv, l2_reg_fc = l2_reg_fc, learning_rate = learning_rate,
	                  D = D, N = N_max, F1 = F1, F2 = F2, F= F, drop_rate = 0.0, ada_rho = rho)
	model.load_weights(join(CURRENT_DIR, "..", "data", "saved_models", "GNN", "saved_model_GNN_best_hyperparameters"))
	get_fingerprint_fct = K.function([model.layers[0].input, model.layers[26].input,
                                  model.layers[3].input, model.layers[36].input],
                                  [model.layers[-10].output])
	return(get_fingerprint_fct)

def get_metabolite_representations(df, get_fingerprint_fct):
    df["GNN rep"] = ""
    i = 0
    n = len(df)
    
    cid_all = df["ID"]
    
    while i*64 <= n:
        if (i+1)*64  <= n:
            XE, X, A, extras = get_representation_input(cid_all[i*64:(i+1)*64])
            representations = get_fingerprint_fct([np.array(XE), np.array(X),np.array(A),
                                                   np.array(extras)])[0]
            df["GNN rep"][i*64:(i+1)*64] = list(representations[:, :52])
        else:
            XE, X, A, extras = get_representation_input(cid_all[i*64:])
            representations = get_fingerprint_fct([np.array(XE), np.array(X),np.array(A), 
                                                   np.array(extras)])[0]
            df["GNN rep"][i*64:] = list(representations[:, :52])
        i += 1
    return(df)


def get_representation_input(cid_list):
    XE = ();
    X = ();
    A = ();
    UniRep = ();
    extras = ();
    # Generate data
    for cid in cid_list:
        X = X + (np.load(join(metabolite_dir, "GNN_input_data", cid + '_X.npy')), );
        XE = XE + (np.load(join(metabolite_dir, "GNN_input_data", cid + '_XE.npy')), );
        A = A + (np.load(join(metabolite_dir, "GNN_input_data", cid + '_A.npy')), );
        extras =  extras + (np.load(join(metabolite_dir, "GNN_input_data", cid + '_extras.npy')), );
    return(XE, X, A, extras)

# Model parameters
F1 = 32         # feature dimensionality of atoms
F2 = 10         # feature dimensionality of bonds
F = F1+F2



class Linear(layers.Layer):

    def __init__(self, dim=(1,1,42,64)):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value = w_init(shape=(dim),
                                                  dtype='float32'),
                             trainable=True)
    def call(self, inputs):
        return tf.matmul(inputs, self.w)
    
    
class Linear_with_bias(layers.Layer):

    def __init__(self, dim):
        super(Linear_with_bias, self).__init__()
        w_init = tf.random_normal_initializer()
        b_init = tf.constant_initializer(0.1)
        self.w = tf.Variable(initial_value = w_init(shape=(dim),
                                                  dtype='float32'),
                             trainable=True)
        self.b = tf.Variable(initial_value = b_init(shape=[self.w.shape[-1]], dtype='float32'), trainable=True)
        
    def call(self, inputs):
        return tf.math.add(tf.matmul(inputs, self.w), self.b)


def DMPNN(l2_reg_conv, l2_reg_fc, learning_rate, D, N, F1, F2, F, drop_rate = 0.15, ada_rho = 0.95):

    # Model definition
    XE_in = Input(shape=(N, N, F), name = "XE", dtype='float32')
    X_in = Input(shape=(N, F1), dtype='float32')
    Extras_in = Input((2), name ="Extras", dtype='float32')

    X = tf.reshape(X_in, (-1, N, 1, F1))
    A_in = Input((N, N, 1),name ="A", dtype='float32') # 64 copies of A stacked behind each other
    Wi = Linear((1,1,F,D))
    Wm1 = Linear((1,1,D,D))
    Wm2= Linear((1,1,D,D))
    Wa = Linear((1,D+F1,D))

    W_fc1 = Linear_with_bias([D + 2, 32])
    W_fc2 = Linear_with_bias([32, 16])
    W_fc3=  Linear_with_bias([16, 1])

    OnesN_N = tf.ones((N,N))
    Ones1_N = tf.ones((1,N))

    H0 = relu(Wi(XE_in)) #W*XE

    #only get neighbors in each row: (elementwise multiplication)
    M1 = tf.multiply(H0, A_in)
    M1 = tf.transpose(M1, perm =[0,2,1,3])
    M1 = tf.matmul(OnesN_N, M1)
    M1 = add(inputs= [M1,-tf.transpose(H0, perm =[0,2,1,3])])
    M1 = tf.multiply(M1, A_in)
    H1 = add(inputs = [H0, Wm1(M1)])
    H1 = relu(BatchNormalization(momentum=0.90, trainable=True)(H1))

    M2 = tf.multiply(H1, A_in)
    M2 = tf.transpose(M2, perm =[0,2,1,3])
    M2 = tf.matmul(OnesN_N, M2)
    M2 = add(inputs= [M2,-tf.transpose(H1, perm =[0,2,1,3])])
    M2 = tf.multiply(M2, A_in)
    H2 = add(inputs = [H0, Wm2(M2)]) 
    H2 = relu(BatchNormalization(momentum=0.90, trainable=True)(H2))
    
    M_v = tf.multiply(H2, A_in)
    M_v = tf.matmul(Ones1_N, M_v)
    XM = Concatenate()(inputs= [X, M_v])
    H = relu(Wa(XM))
    h = tf.matmul(Ones1_N, tf.transpose(H, perm= [0,2,1,3]))
    h = tf.reshape(h, (-1,D))
    h_extras = Concatenate()(inputs= [h, Extras_in])
    h_extras = BatchNormalization(momentum=0.90, trainable=True)(h_extras)

    fc1 = relu(W_fc1(h_extras))
    fc1 = BatchNormalization(momentum=0.90, trainable=True)(fc1)
    fc1 = Dropout(drop_rate)(fc1)

    fc2 =relu(W_fc2(fc1))
    fc2 = BatchNormalization(momentum=0.90, trainable=True)(fc2)

    output = W_fc3(fc2)
    
    def total_loss(y_true, y_pred):
        reg_conv_loss = (tf.nn.l2_loss(Wi.w) + tf.nn.l2_loss(Wm1.w)+ tf.nn.l2_loss(Wm2.w) + tf.nn.l2_loss(Wa.w))
        reg_fc_loss = (tf.nn.l2_loss(W_fc1.w) +tf.nn.l2_loss(W_fc2.w) +tf.nn.l2_loss(W_fc3.w))
        mse_loss = tf.keras.losses.MSE(y_true, y_pred)
        return(tf.reduce_mean(mse_loss + l2_reg_conv * reg_conv_loss + l2_reg_fc * reg_fc_loss))

    # Build model
    model = Model(inputs=[XE_in, X_in, A_in, Extras_in], outputs=output)

    #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, H1_batch.updates)
    optimizer = Adadelta(lr=learning_rate, rho = ada_rho)

    model.compile(optimizer=optimizer, loss=total_loss, metrics=['mse', "mae"])
    return(model)





