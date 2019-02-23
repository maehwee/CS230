'''
This code is a first go at training NNs to study a phase transition in the VBS model.
Authors: Chao Wang, Mae Teo
'''

import numpy as np
from keras import layers
from keras.regularizers import l1, l2, l1_l2
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')




'''
Functions to extract data and make labels for training and test sets, in the right shape.
'''


def assign_kin_mat(lx, jmat, shift_x, shift_y, dx, dy, step_x, step_y, bond_to_xy, bonds_to_sites, index):
    jp = -1
    for jx in range((0+shift_x),lx,step_x):
        for jy in range((0+shift_y),lx, step_y):
            jp = jp+1
            jx1 = jx
            jx2 = (jx+dx)%lx
            jy1 = jy
            jy2 = (jy+dy)%lx
            bond_to_xy[jmat, jp, 0] = float(jx1+jx2)/2.0 #x coordinate of bond position
            bond_to_xy[jmat, jp, 1] = float(jy1+jy2)/2.0 #y coordinate of bond position
            bonds_to_sites[0, jmat, jp] = index[jx1, jy1] #site 1 j
            bonds_to_sites[1, jmat, jp] = index[jx2, jy2] #site 2 j

def set_up_data_X(h_value, no_tau = 0):
    """
    Assuming file path to data is fixed, given the value of h (a STRING not float),
    returns X dataset (without Y labels) in our desired format (m, L*L, L*L, 3), shuffled,
    where 
    channel 1 = real component of Gij,
    channel 2 = imaginary component of Gij,
    channel 3 = value of tau_ij; we have not decided if this channel will go or stay
    """
    # these are parameters for the simulations: they may not be used below
    lx = 8 # the system is on a lx-by-lx lattice
    ntau = 120 # number of time slices in each sample
    #hfile = 2.27 # h value
    nwrite = 200 # we write out samples once every nwrite sampling

    # load Green's function file
    filename_g = '../Data/add_spin_i_cnn/qsl_size=8_beta=12.0_u=-3._alpha=-1.5_h=' + h_value + '_fz=0_dtau=0.1_isrand=1_v=2.0_v1=2.0/seed=18/test_G.out'
    filename_tau = '../Data/add_spin_i_cnn/qsl_size=8_beta=12.0_u=-3._alpha=-1.5_h=' + h_value + '_fz=0_dtau=0.1_isrand=1_v=2.0_v1=2.0/seed=18/test_tau.out'
    data_g_temp = np.fromfile(filename_g, dtype=np.float64)
    single = 2*lx**4

    data_g_temp = np.reshape(data_g_temp, [single,-1], order='F')
    data_g = np.copy(data_g_temp)

    # Reshape Green's function properly
    data_g = np.reshape(data_g, [2,lx**2,lx**2,-1], order='F') # data_g dimensions: re/im parts (2), j1 (lx**2), j2 (lx**2), sample #
    data_g = np.transpose(data_g)
    
    if no_tau == 1:
        X = data_g
        np.random.shuffle(X)
        print("The shape of the data without tau is:")
        print(X.shape)
        return X
        
        
    # define indexing conventions for all the sites
    ind = 0
    index = np.zeros((lx,lx),dtype=int)
    for jy in range(lx):
        for jx in range(lx):
            index[jx, jy] = ind
            ind = ind+1

    
    # load tau (pseudospins, which take values + or - 1, which live on the bonds between nearest neighbours)
    data_tau = np.fromfile(filename_tau, dtype=np.int8)
    num_pair = lx**2//2
    print("Data tau raw shape:")
    print(data_tau.shape)
    
    data_tau = np.reshape(data_tau, [ntau, num_pair, 4, -1], order='F') 
    num_sample = data_tau.shape[3]
    # jtau, jp, jmat, sample #: don't bother understanding what jp and jmat are 
    # (I will map jp, jmat to the indices of the two sites)
    # but jtau is index of imaginary-time slice
    #data_tau = np.transpose(data_tau) #reverse index order
    #print(data_tau.shape)

    # define indexing conventions for all the bonds between nearest neighbor sites
    bonds_to_sites = np.zeros((2, 4, num_pair), dtype=int) # j for site 1/2, jmat, jp
    bond_to_xy = np.zeros((4, num_pair, 2), dtype=float)
    #(l, jmat, shift_x, shift_y, dx, dy, step_x, step_y, bond_to_xy)
    assign_kin_mat(lx, 0, 0, 0, 1, 0, 2, 1, bond_to_xy, bonds_to_sites, index)
    assign_kin_mat(lx, 1, 1, 0, 1, 0, 2, 1, bond_to_xy, bonds_to_sites, index)
    assign_kin_mat(lx, 2, 0, 0, 0, 1, 1, 2, bond_to_xy, bonds_to_sites, index)
    assign_kin_mat(lx, 3, 0, 1, 0, 1, 1, 2, bond_to_xy, bonds_to_sites, index)


    # Transform the tau data into indexing by sites instead of (jmat, jp)
    data_tau_new = np.zeros((lx**2, lx**2, num_sample)) #jsite1, jsite2, jsample
    jtau = ntau//2 - 1 #pick out the jtau that corresponds to the time slice we used for equal-time Green's function
    for jmat in range(4):
        for jp in range(num_pair):
            j1 = int(bonds_to_sites[0, jmat, jp])
            j2 = int(bonds_to_sites[1, jmat, jp])
            data_tau_new[j1, j2, :] = data_tau[jtau, jp, jmat, :]
            data_tau_new[j2, j1, :] = data_tau[jtau, jp, jmat, :]
        

    # concatenate data_g, data_tau_new
    data_tau_new = np.transpose(np.expand_dims(data_tau_new,axis=0))    
    X = np.concatenate((data_g, data_tau_new), axis = 3) 
    np.random.shuffle(X)
    print("The shape of the data is:")
    print(X.shape)
    
    return X


def set_up_labels_Y(list_of_X, list_of_Y, list_of_each = []):
    """
    Create Y labels.
    IF no argument given to list_of_each,
    Given function input ([X1, X2], [0,1]), we return labels Y = [0 0 0 ... 1 1] with the lengths of X1, X2.
    If list_of_each is given, eg [200,300], total length will be 200 + 300.
    """
    Y = np.zeros((0,1))
    for i in range(len(list_of_X)):
        if list_of_each == []:
            new = np.full((list_of_X[i].shape[0], 1), list_of_Y[i])
        else:
            new = np.full((list_of_each[i], 1), list_of_Y[i])
        Y = np.concatenate((Y, new))
    print('The shape of the combined labels Y is:')
    print(Y.shape)
    return Y
        
def concatenate_X_data(list_of_X, list_of_each = [], list_starting_index = []):
    """
    Combine X data. If list_of_each argument is no given, use all data.
    If not, list_of_each[200, 200] means use 200 from each dataset and combine them.
    list_starting_index = [5,5] means start with data from 5 on war
    """
    if list_of_each == []:
        X_combined = np.concatenate([*list_of_X], axis = 0)
    else:
        list_of_X_wanted = []
        for i in range(len(list_of_X)):
            list_of_X_wanted.append(list_of_X[i][list_starting_index[i]:list_starting_index[i]+list_of_each[i]])
        X_combined = np.concatenate([*list_of_X_wanted], axis = 0)
    print('The shape of the combined data X is:')
    print(X_combined.shape)
    return X_combined


'''
Functions to test predictions of NN.
'''


def evaluate_diff_h(modelname):
    '''
    evaluate test accuracy on h = 2.28 - 2.34 test sets
    '''
    accs = []
    for i in range(28, 35):
        print('h = 2.' + str(i))
        preds1=modelname.evaluate(x = h_testing['x_test_'+ str(i)], y = h_testing['y_test_'+ str(i)])
        print ("Test Accuracy = " + str(preds1[1]))
        accs.append(preds1[1])
        print()
    print(accs) 
    

'''
Set up training and testing data.
'''
# set up data from each h to be of shape (m_h, 64, 64, 2)
x27_notau = set_up_data_X('2.27',1)
x28_notau = set_up_data_X('2.28',1)
x29_notau = set_up_data_X('2.29',1)
x30_notau = set_up_data_X('2.30',1)
x31_notau = set_up_data_X('2.31',1)
x32_notau = set_up_data_X('2.32',1)
x33_notau = set_up_data_X('2.33',1)
x34_notau = set_up_data_X('2.34',1)
x35_notau = set_up_data_X('2.35',1)

#choose training data, from largest and smallest h
x_train = concatenate_X_data([x27_notau, x35_notau],[2500,2500], [0,0]) #use 2500 examples from each
y_train = set_up_labels_Y([x27_notau, x35_notau],[0,1],[2500,2500])

#testing data (from the same distribution)
x_test = concatenate_X_data([x27_notau, x35_notau],[385,385],[2500,2500]) #use data from index 2500 - 2885
y_test = set_up_labels_Y([x27_notau, x35_notau],[0, 1],[385,385]) 

#set up test data from intermediate h values
print('test set with various intermediate h values')
h_testing={}
h_testing['y_test_28'] = set_up_labels_Y([x28_notau],[1],[x28_notau.shape[0] - 1000]) #testing
h_testing['y_test_29'] = set_up_labels_Y([x29_notau],[1],[x29_notau.shape[0] - 1000]) #testing
h_testing['y_test_30'] = set_up_labels_Y([x30_notau],[1],[x30_notau.shape[0] - 1000]) #testing
h_testing['y_test_31'] = set_up_labels_Y([x31_notau],[1],[x31_notau.shape[0] - 1000]) #testing
h_testing['y_test_32'] = set_up_labels_Y([x32_notau],[1],[x32_notau.shape[0] - 1000]) #testing
h_testing['y_test_33'] = set_up_labels_Y([x33_notau],[1],[x33_notau.shape[0] - 1000]) #testing
h_testing['y_test_34'] = set_up_labels_Y([x34_notau],[1],[x34_notau.shape[0] - 1000]) #testing


h_testing['x_test_28'] = concatenate_X_data([x28_notau],[x28_notau.shape[0] - 1000],[1000]) #testing
h_testing['x_test_29'] = concatenate_X_data([x29_notau],[x29_notau.shape[0] - 1000],[1000]) #testing
h_testing['x_test_30'] = concatenate_X_data([x30_notau],[x30_notau.shape[0] - 1000],[1000]) #testing
h_testing['x_test_31'] = concatenate_X_data([x31_notau],[x31_notau.shape[0] - 1000],[1000]) #testing
h_testing['x_test_32'] = concatenate_X_data([x32_notau],[x32_notau.shape[0] - 1000],[1000]) #testing
h_testing['x_test_33'] = concatenate_X_data([x33_notau],[x33_notau.shape[0] - 1000],[1000]) #testing
h_testing['x_test_34'] = concatenate_X_data([x34_notau],[x34_notau.shape[0] - 1000],[1000]) #testing


"""
Convolutional Neural Network: define, compile, train, test
"""
def NewTestModel(input_shape):
    """
    Single conv layer.
    """
    X_input = Input(input_shape)
    X = X_input

    # CONV -> BN -> RELU Block
    X = Conv2D(32, (3, 3), strides = (2,2), padding='same', kernel_regularizer=l1_l2(l1=0.00, l2=0.03), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    # MAXPOOL
    X = MaxPooling2D((6, 6), name='max_pool0')(X)
    # FC
    X = Flatten()(X)
    X = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=0.0, l2=0.03), name='fc0')(X)
    #SIGMOID
    X = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=0.0, l2=0.03), name='fc1')(X)

    model = Model(inputs = X_input, outputs = X, name='NewTestModel')
    return model

# summarize and compile model
newtestmodel = NewTestModel((64,64,2))
print("Total number of parameters:")
print(newtestmodel.count_params())
newtestmodel.summary()
newtestmodel.compile(optimizer = Adam(lr=0.001), loss = "binary_crossentropy", metrics = ["accuracy"])

#train for 20 epochs, printing both training and dev set accuracy in each step; use early stopping
for i in range(20):
    print()
    print()
    print('Epoch: ')
    print(i)
    newtestmodel.fit(x = x_train, y = y_train, epochs = 1, batch_size = 200, shuffle = True)
    preds = newtestmodel.evaluate(x = x_test, y = y_test)
    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    

#evaluate on intermediate values of h
evaluate_diff_h(newtestmodel)

'''
Logistic regression: define, compile, train, test
'''
def simplestModelwithl2(input_shape):
    """
    Logistic regression
    """
    X_input = Input(input_shape)
    X = X_input
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=0.000, l2=0.03), name='fc')(X)

    model = Model(inputs = X_input, outputs = X, name='simplestModelwithl2')
    return model

simplestmodelwithl2 = simplestModelwithl2((64,64,2))
simplestmodelwithl2.compile(optimizer = Adam(lr=0.01), loss = "binary_crossentropy", metrics = ["accuracy"])
simplestmodelwithl2.summary()

# summarize and compile model
simplestmodelwithl2.fit(x = x_train, y = y_train, epochs = 5, batch_size = 200, shuffle = True)
preds = simplestmodelwithl2.evaluate(x = x_test, y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#evaluate on intermediate values of h
evaluate_diff_h(simplestmodelwithl2)   

