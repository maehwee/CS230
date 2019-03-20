#!/usr/bin/env python
# coding: utf-8

import os
import cPickle as pickle
import numpy as np
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pylab as plt
from src.TTRegression import TTRegression
import urllib
import numpy as np
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

'''
This code is a first go at training NNs to study phase transition in the VBS model.
Authors: Chao Wang, Mae Teo
'''

'''
Functions to extract and transform data in the right shape.
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

def sites_indices(lx):
    # define indexing conventions for all the sites
    ind = 0
    index = np.zeros((lx,lx),dtype=int)
    for jy in range(lx):
        for jx in range(lx):
            index[jx, jy] = ind
            ind = ind+1
    return index

def latt_dist(jx1, jx2, jy1, jy2, lx):
    dist_x = float(min(abs(jx1-jx2), lx-abs(jx1-jx2)))
    dist_y = float(min(abs(jy1-jy2), lx-abs(jy1-jy2)))
    return np.sqrt(dist_x**2+dist_y**2)


def g_make_local(data_g, lx, max_dist = 1.42):
    data_g_new = np.zeros((data_g.shape[0], data_g.shape[1]*data_g.shape[2])) #data_g.shape[0] is number of samples
    count = 0
    index = sites_indices(lx)
    for jx1 in range(lx):
        for jy1 in range(lx):
            for jx2 in range(lx):
                for jy2 in range(lx):
                    if latt_dist(jx1, jx2, jy1, jy2, lx) < max_dist:
                        ind1 = index[jx1, jy1]
                        ind2 = index[jx2, jy2]
                        if ind1 <= ind2:
                            data_g_new[:, count] = data_g[:, ind1, ind2]
                            count += 1
    data_g_new = data_g_new[:,:count]
    print "count=", count
    return data_g_new

def g_translate(data_g, lx):
    data_g_new = data_g.copy()
    dx = np.random.randint(0,lx)
    dy = np.random.randint(0,lx)
    ind_to_indt = [0]*lx**2
    inds = sites_indices(lx)
    for jx in range(lx):
        for jy in range(lx):
            ind = inds[jx, jy]
            ind_to_indt[ind] = inds[(jx+dx)%lx, (jy+dy)%lx]
    for ind1 in range(lx**2):
        for ind2 in range(lx**2):
            ind1_translate = ind_to_indt[ind1]
            ind2_translate = ind_to_indt[ind2]
            data_g_new[:, ind1, ind2] = data_g[:, ind1_translate, ind2_translate]
    return data_g_new

def rotate1(dataset_name, lx):
    """
    Rotate in sample space 90 degrees.
    """
    reshaped = dataset_name.reshape((dataset_name.shape[0], lx, lx, lx, lx))
    reshaped_rotated = np.zeros(reshaped.shape)
    for example in range(reshaped.shape[0]):
        for row in range(lx):
            for col in range(lx):
                for i in range(lx):
                    for j in range(lx):
                        new_x = (row -(j - col))%lx
                        new_y = (col + (i - row))%lx
                        reshaped_rotated[example, row, col, i, j] = reshaped[example, row, col, new_x, new_y]
    original_shape_new = reshaped_rotated.reshape((reshaped.shape[0], lx**2, lx**2))
    return original_shape_new

def set_up_data_X(h_value, lattice_side, no_tau = 0, filename1 = '/qsl_size=8_beta=12.0_u=-3._alpha=-1.5_h=', filename2 = '_fz=0_dtau=0.1_isrand=1_v=2.0_v1=2.0/seed=18/'):
    """
    Assuming file path to data is fixed, given the value of foldername (a string) and h (a STRING not float),
    returns X dataset (without Y labels) in our desired format (m, L*L, L*L, 3), shuffled,
    where
    channel 1 = real component of Gij,
    channel 2 = imaginary component of Gij,
    channel 3 = value of tau_ij; we have not decided if this channel will go or stay
    """
    # these are parameters for the simulations: they may not be used below
    lx = lattice_side # the system is on a lx-by-lx lattice
    ntau = 120 # number of time slices in each sample
    #hfile = 2.27 # h value
    nwrite = 200 # we write out samples once every nwrite sampling

    # load Green's function file
    filename_g = '../../Data/add_spin_cnn' + filename1 + h_value + filename2 + 'test_G.out'
    filename_tau = '../../Data/add_spin_cnn' + filename1 + h_value + filename2 + 'test_tau.out'
    data_g_temp = np.fromfile(filename_g, dtype=np.float64)
    single = 2*lx**4

    print("raw data shape is " + str(data_g_temp.shape))

    if data_g_temp.shape[0] % single != 0:
        number_of_examples = data_g_temp.shape[0]//single
        data_g_temp = data_g_temp[0:single * number_of_examples]

    data_g_temp = np.reshape(data_g_temp, [single,-1], order='F')
    data_g = np.copy(data_g_temp)

    # Reshape Green's function properly
    data_g = np.reshape(data_g, [2,lx**2,lx**2,-1], order='F') # data_g dimensions: re/im parts (2), j1 (lx**2), j2 (lx**2), sample #
    data_g = np.transpose(data_g)
    data_g = data_g[:,:,:,0] # take only real part

    # Rotations
    data_g_rot1 = rotate1(data_g, lx)
    data_g_rot2 = rotate1(data_g_rot1, lx)
    data_g_rot3 = rotate1(data_g_rot2, lx)

    # Random translations
    np.random.seed(20)
    data_g_trans = g_translate(data_g, lx)

    # Make local
    data_g = g_make_local(data_g, lx)
    data_g_rot1 = g_make_local(data_g_rot1, lx)
    data_g_rot2 = g_make_local(data_g_rot2, lx)
    data_g_rot3 = g_make_local(data_g_rot3, lx)
    data_g_trans = g_make_local(data_g_trans, lx)

    # Combine into augmented dataset
    data_g_aug = np.concatenate((data_g, data_g_rot1, data_g_rot2, data_g_rot3, data_g_trans),axis=0)
    data_g = data_g_aug

    #Throw away long-distance entries of G
    #data_g = g_make_local(data_g, lx) # make dims be (num_samples, m) where m < lx**4 so that TTRegression is happy
    #shp = data_g.shape
    #data_g = np.reshape(data_g, (data_g.shape[0], data_g.shape[1]*data_g.shape[2])) # make dims be (n, lx**2*lx**2) so that TTRegression is happy

    if no_tau == 1:
        X = data_g
        np.random.shuffle(X)
        print("The shape of the data h =  " + h_value + " without tau is:")
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
    print("The shape of the data h =  " + h_value + "is:")
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
            new = np.full((list_of_X[i].shape[0],1), list_of_Y[i])
        else:
            new = np.full((list_of_each[i],1), list_of_Y[i])
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
        X_combined = np.concatenate(list_of_X, axis = 0) #[*list_of_X]
    else:
        list_of_X_wanted = []
        for i in range(len(list_of_X)):
            list_of_X_wanted.append(list_of_X[i][list_starting_index[i]:list_starting_index[i]+list_of_each[i]])
        X_combined = np.concatenate(list_of_X_wanted, axis = 0) #[*list_of_X_wanted]
    print('The shape of the combined data X with is:')
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
        h = 2 + 0.01*i
        print('h = ' + str(h))
        preds1=modelname.evaluate(x = h_testing['x_test_'+ str(i)], y = h_testing['y_test_'+ str(i)])
        print ("Fraction labelled 1 = " + str(preds1[1]))
        accs.append((h,preds1[1]))
        print()
    for a,b in accs:
        print(a,b)


def evaluate_diff_h_2(model_name, threeD = 0, wantPlot = True, norm = False):
    '''
    evaluate test accuracy on h = 1.7 - 2.7 test sets
    '''
    accs = []
    if threeD == 0:
        labl = ""
    else:
        labl = str("_3d")
    for i in range(1, 7):
        h = 1.5+0.2*i
        print('h = ' + str(h))
        if norm == False:
            preds1=model_name.evaluate(x = h_testing['x_test_'+ str(i)+labl], y = h_testing['y_test_'+ str(i)])
        else:
            preds1=model_name.evaluate(x = h_testing['x_norm_'+ str(h)], y = h_testing['y_test_'+ str(i)])
        print ("Fraction labelled 1 = " + str(preds1[1]))
        accs.append((h,preds1[1]))
        print()
    for a,b in accs:
        print(a,b)
    if wantPlot == True:
        # plot benchmark
        x1,y1 = np.transpose(normalized_trend_of_mean())
        plt.scatter(x1,y1, c='g', marker = 'x')
        # plot results
        x,y = np.transpose(accs)
        plt.scatter(x,y, c = 'b')
        plt.title("fraction labelled 1 vs. h")

# loading data
# size 8
dataset_side8 = {}
could_not_load = []
for h_value in ['1.5', '1.7', '1.9', '2.1', '2.2', '2.3', '2.5', '2.7', '2.9'
               ]:
    print("side 8, h = " + h_value)
    try:
        dataset_side8[h_value] = set_up_data_X(h_value,8,1,'/qsl_size=8_beta=12.0_u=-3._alpha=-1.5_h=','_fz=0_dtau=0.1_isrand=1_v=2.0_v1=2.0/seed=18/') #notau = 1
    except Exception as e:
        print("ERROR: Cannot load properly!")
        print(e)
        could_not_load.append(h_value)
        pass
    print()
print("dataset_side8, COULD NOT LOAD h = " + str(could_not_load))

print()
print()
print("----------------------------------------------------------")

# setting up training and validation sets

val_size = 2000
h_0 = '1.5'
h_1 = '2.9'

#choose training data, from largest and smallest h
x_train = concatenate_X_data([dataset_side8[h_0], dataset_side8[h_1]],[dataset_side8[h_0].shape[0] - val_size, dataset_side8[h_1].shape[0] - val_size], [0,0]) #use  all but 500 examples from each
y_train = set_up_labels_Y([dataset_side8[h_0], dataset_side8[h_1]],[0,1],[dataset_side8[h_0].shape[0] - val_size, dataset_side8[h_1].shape[0] - val_size])
y_train = np.squeeze(y_train)

#validation data (from the same distribution)
x_val = concatenate_X_data([dataset_side8[h_0], dataset_side8[h_1]],[val_size,val_size],[dataset_side8[h_0].shape[0] - val_size,dataset_side8[h_1].shape[0] - val_size])
y_val = set_up_labels_Y([dataset_side8[h_0], dataset_side8[h_1]],[0, 1],[val_size,val_size])
y_val = np.squeeze(y_val)

# setting up tensor train model and train it

plain_sgd = {}
riemannian_sgd = {}

#for batch_size in [-1, 100, 500]:

batch_size = 100

# To use the same order of looping through objects for all runs.
np.random.seed(0)
model = TTRegression('all-subsets', 'logistic', 4, 'sgd', max_iter=30, verbose=3,
                     fit_intercept=False, batch_size=batch_size, reg=0.005)
print "OK1"
model.fit_log_val(x_train, y_train, x_val, y_val)
print "OK2"
plain_sgd[batch_size] = model

# np.random.seed(0)
# # To use the same order of looping through objects for all runs.
# rieamannian_model = TTRegression('all-subsets', 'logistic', 4, 'riemannian-sgd', max_iter=20, verbose=1,
#                                  batch_size=batch_size, fit_intercept=False, reg=0.)
# rieamannian_model.fit_log_val(x_train, y_train, x_val, y_val)
# riemannian_sgd[batch_size] = rieamannian_model

# look at validation accuracy

model1 = plain_sgd[batch_size]
A = model1.predict_log_proba(x_val)
#print A[900:]

#validation set accuracy
wrong = 0
for i in range(A.shape[0]):
    pred = 0
    if A[i][0]>=A[i][1]:
        pred = 0
    else:
        pred = 1
    if pred != y_val[i]:
        wrong += 1
print "validation set accuracy=", 1.0-float(wrong)/float(A.shape[0])

#print "Ok1"
#model2 = riemannian_sgd[batch_size]
#model2.predict_proba(x_val)


# use the tensor train model to predict on intermediate h values
print('test set with various intermediate h values')

h_testing_side8={}
h_testing_side8_labels={}

for h, data in dataset_side8.items():
    print('dataset_side8, h = ' + h)
    h_testing_side8[h] = concatenate_X_data([data],[data.shape[0]],[0])
    h_testing_side8_labels[h] = set_up_labels_Y([data],[1],[data.shape[0]])

    print

# compute predictions and plot them
hs = []
preds = []
for h, data in dataset_side8.items():
    print('dataset_side8, h = ' + h)
    h_testing_side8[h] = concatenate_X_data([data],[data.shape[0]],[0])
    h_testing_side8_labels[h] = set_up_labels_Y([data],[1],[data.shape[0]])

    A = model1.predict_log_proba(h_testing_side8[h])
    pred = 0
    for i in range(A.shape[0]):
        if A[i][0]>=A[i][1]:
            pred += 0
        else:
            pred += 1
    pred = pred/float(A.shape[0])
    hs.append(float(h))
    preds.append(pred)
    print "test set prediction=", pred

print hs
print preds
print
plt.scatter(hs, preds, c='r')
