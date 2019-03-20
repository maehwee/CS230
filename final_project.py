'''
This code sets up datasets and uses Logistic Regression and a ConvNet to study phase transitions in the VBS model.
Authors: Chao Wang, Mae Teo
'''

import numpy as np
from PIL import Image
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras import layers
from keras.regularizers import l1, l2, l1_l2
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv3D
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
    filename_g = '../Data/add_spin_cnn' + filename1 + h_value + filename2 + 'test_G.out'
    filename_tau = '../Data/add_spin_cnn' + filename1 + h_value + filename2 + 'test_tau.out'
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
    shp = data_g.shape
    data_g = np.reshape(data_g, (*shp, 1)) # make dims be (n, lx**2, lx**2, 1) so that ConvLayer is happy
    
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
    print('The shape of the combined data X with is:')
    print(X_combined.shape)
    return X_combined
        

# data augmentation --- each sample can be rotated 90, 180, 270 degrees

def rotate1(dataset_name, lx):
    """
    Rotate 90 degrees in sample space.
    """
    reshaped = dataset_name.reshape((dataset_name.shape[0], lx, lx, lx, lx, 1))
    reshaped_rotated = np.zeros(reshaped.shape)
    for example in range(reshaped.shape[0]):
        for row in range(lx):
            for col in range(lx):
                for i in range(lx):
                    for j in range(lx):
                        new_x = (row -(j - col))%lx
                        new_y = (col + (i - row))%lx
                        reshaped_rotated[example, row, col, i, j, 0] = reshaped[example, row, col, new_x, new_y, 0]
    original_shape_new = reshaped_rotated.reshape((reshaped.shape[0], lx**2, lx**2, 1))
    return original_shape_new




"""
Functions to analyze data
"""

# visualize each training example (or average over examples) using matshow

def visualize_data(lx, row, col, dataset, h_value):
    """
    Prints the correlation at site (row, col) in image form.
    
    h_value = string representing value of h
    lx = length of lattice side
    row, col = coordinates whose Greens function we want to look at
    dataset = dictionary that stores Greens functions for various h's
    
    """
    
    # print full data, show mean over examples
    avg_over_samples = np.sum(dataset[h_value], axis = 0)/dataset[h_value].shape[0]
    plt.matshow(avg_over_samples[:,:,0])
    plt.title('side = ' + str(lx) + ', h = ' + h_value)
    plt.savefig('./plots/training_example_augmented/qsldata/size=' + str(lx) + "_h=" + h_value + "full_example_" + "mean" + ".png")
    
      # reshaped to look at one site's correlation function
    reshaped = dataset[h_value].reshape((dataset[h_value].shape[0], lx, lx, lx, lx, 1))
    
     #print 4 examples
    plt.matshow(reshaped[5,row,col,:,:,0])
    plt.title('side = ' + str(lx) + ', h = ' + h_value + ' at site '  + str(row) + "," +  str(col))
    plt.savefig('./plots/training_example_augmented/qsldata/size=' + str(lx) + "_h=" + h_value + "_example_" + "1" + ".png")
    
    plt.matshow(reshaped[10,row,col,:,:,0])
    plt.title('side = ' + str(lx) + ', h = ' + h_value + ' at site '  + str(row) + "," +  str(col))
    plt.savefig('./plots/training_example_augmented/qsldata/size=' + str(lx) + "_h=" + h_value + "_example_" + "2" + ".png")
    
    plt.matshow(reshaped[15,row,col,:,:,0])
    plt.title('side = ' + str(lx) + ', h = ' + h_value + ' at site '  + str(row) + "," +  str(col))
    plt.savefig('./plots/training_example_augmented/qsldata/size=' + str(lx) + "_h=" + h_value + "_example_" + "3" + ".png")
    
    plt.matshow(reshaped[12001,row,col,:,:,0])
    plt.title('side = ' + str(lx) + ', h = ' + h_value + ' at site '  + str(row) + "," +  str(col))
    plt.savefig('./plots/training_example_augmented/qsl/size=' + str(lx) + "_h=" + h_value + "_example_" + "4" + ".png")
    
    #print mean over examples
    avg_over_samples = np.sum(dataset[h_value], axis = 0)/dataset[h_value].shape[0]
    avg_over_samples_reshaped = avg_over_samples.reshape((lx,lx,lx,lx))
    plt.matshow(avg_over_samples_reshaped[row,col,:,:]) # real part
    plt.title('side = ' + str(lx) + ', h = ' + h_value + ' at site '  + str(row) + "," +  str(col))
    plt.savefig('./plots/training_example_augmented/qsldata/size=' + str(lx) + "_h=" + h_value + "_example_" + "mean" + ".png")


    
    
def trend_of_mean(dataset_name, smallest = 1.5, largest = 2.9):
    """
    Finds the sum of all elements in each example, averaged over all examples for each h.
    Also calculates the normalized value, such that h = smallest has mean 0, h = largest has mean 1.
    This is to compare to the performance of NNs.
    """
    to_be_plotted = []
    smallest_mean = np.mean(dataset_name[str(smallest)])
    largest_mean = np.mean(dataset_name[str(largest)])
    for h, examples in dataset_name.items():
        avg = np.mean(examples)
        normalized_avg = (avg - smallest_mean)/(largest_mean - smallest_mean)
        to_be_plotted.append([float(h), normalized_avg])
        print(h, normalized_avg)
    return to_be_plotted

            
            
def plot_this(list_of_tuples, color = 'r', title = ''):
    """
    Easily plot a list of [a,b] (where a and b are floats) with this function.
    """
    x,y = np.transpose(list_of_tuples)
    plt.scatter(x,y, c=color)
    plt.title(title)
    
def plot_this_line(list_of_tuples, style = 'ro', title = ''):
    """
    Easily plot a list of [a,b] (where a and b are floats) with this function.
    """
    x,y = np.transpose(list_of_tuples)
    plt.plot(x,y, style)
    plt.title(title)
      
def evaluate_diff_h_3(model_name, test_set_dict, labels_dict, list_of_test_data_keys):
    """
    Input the model and datasets we want to make predictions for, output a list of tuples
    of the "fraction labelled 1" for each dataset.
    """
    table_of_values = []
    for test_data_key in list_of_test_data_keys:
        print("Evaluating on dataset: " + test_data_key)
        preds1=model_name.evaluate(x = test_set_dict[test_data_key], y = labels_dict[test_data_key])
        print ("Fraction labelled 1 = " + str(preds1[1])) 
        table_of_values.append([float(test_data_key), preds1[1]])
        print()
        print()
    print(table_of_values)
    return(table_of_values)
    
    
    
'''
Load and set up the data
'''
# Use dictionaries to set up data of each h to be of shape (m_h, lx**2, lx**2, 1)


# size 8
dataset_side8 = {}
could_not_load = []
for h_value in ['1.5','1.7','1.9','2.1','2.2','2.3','2.5','2.7','2.9',
               ]:
    print("side 8, h = " + h_value)
    try:
        dataset_side8[h_value] = set_up_data_X(h_value,8,1,'/qsl_size=8_beta=12.0_u=-3._alpha=-1.5_h=','_fz=0_dtau=0.1_isrand=1_v=2.0_v1=2.0/seed=18/') #notau = 1
    except:
        print("ERROR: Cannot load properly!")
        could_not_load.append(h_value)
        pass
    print()
print("dataset_side8, COULD NOT LOAD h = " + str(could_not_load))
    
print()
print()
print("----------------------------------------------------------")
    
    
# size 10
dataset_side10 = {}
could_not_load = []
for h_value in ['1.5','1.7','1.9','2.1','2.2','2.3','2.5','2.7','2.9',
               ]:
    print("side 10, h = " + h_value)
    try:
        dataset_side10[h_value] = set_up_data_X(h_value,10,1,'/qsl_size=10_beta=12.0_u=-3._alpha=-1.5_h=','_fz=0_dtau=0.1_isrand=1_v=2.0_v1=2.0/seed=18/') #notau = 1
    except:
        print("ERROR: Cannot load properly!")
        could_not_load.append(h_value)
        pass
    print()
print("dataset_side10, COULD NOT LOAD h = " + str(could_not_load))
    
print()
print()
print("----------------------------------------------------------")




# size 12
dataset_side12 = {}
could_not_load = []
for h_value in ['1.5','1.7','1.9','2.1','2.2','2.3','2.5','2.7','2.9',
               ]:
    print("side 12, h = " + h_value)
    try:
        dataset_side12[h_value] = set_up_data_X(h_value,12,1,'/qsl_size=12_beta=12.0_u=-3._alpha=-1.5_h=','_fz=0_dtau=0.1_isrand=1_v=2.0_v1=2.0/seed=18/') #notau = 1
    except:
        print("ERROR: Cannot load properly!")
        could_not_load.append(h_value)
        pass
    print()
print("dataset_side12, COULD NOT LOAD h = " + str(could_not_load))
    
print()
print()
print("----------------------------------------------------------")


# QSL. size 8,beta = 5, widely spaced apart h
dataset_qsl_beta5 = {}
could_not_load = []
for h_value in ['0.05','0.15','0.25', '0.1','0.2','0.3', '0.4', '0.5', '0.6'
               ]:
    print("qsl beta = 5, h = " + h_value) 
#    try:
    dataset_qsl_beta5[h_value] = set_up_data_X(h_value,8,1,'/qsl_size=8_beta=5.0_u=-3._alpha=-0.51_h=','_fz=0_dtau=0.05_isrand=1_v=0._v1=0.5/seed=18/') #notau = 1
#     except:
#         print("ERROR: Cannot load properly!")
#         could_not_load.append(h_value)
#         pass
    print()
print("COULD NOT LOAD h = " + str(could_not_load))
    
print()
print()
print("----------------------------------------------------------")




# set up rotated datasets, and combine into bigger datasets

# side 8
dataset_side8_1 = {}
dataset_side8_2 = {}
dataset_side8_3 = {}
dataset_side8_augmented = {}

for h_value, data in dataset_side8.items():
    dataset_side8_1[h_value] = rotate1(dataset_side8[h_value], 8)
    dataset_side8_2[h_value] = rotate1(dataset_side8_1[h_value], 8)
    dataset_side8_3[h_value] = rotate1(dataset_side8_2[h_value], 8)
    
    dataset_side8_augmented[h_value] = np.concatenate((dataset_side8[h_value], 
                                                       dataset_side8_1[h_value], 
                                                       dataset_side8_2[h_value], 
                                                    dataset_side8_3[h_value]),
                                                    axis=0)
    dataset_side8_1[h_value] = [] #to not use up so much memory
    dataset_side8_2[h_value] = []
    dataset_side8_3[h_value] = []
    print("size = 8, h = " + h_value + ", shape is ")
    print(dataset_side8_augmented[h_value].shape)

    
set up rotated datasets, and combine into bigger datasets

# side 10
dataset_side10_1 = {}
dataset_side10_2 = {}
dataset_side10_3 = {}
dataset_side10_augmented = {}

for h_value, data in dataset_side10.items():
    dataset_side10_1[h_value] = rotate1(dataset_side10[h_value], 10)
    dataset_side10_2[h_value] = rotate1(dataset_side10_1[h_value], 10)
    dataset_side10_3[h_value] = rotate1(dataset_side10_2[h_value], 10)
    
    dataset_side10_augmented[h_value] = np.concatenate((dataset_side10[h_value], 
                                                       dataset_side10_1[h_value], 
                                                       dataset_side10_2[h_value], 
                                                    dataset_side10_3[h_value]),
                                                    axis=0)
    dataset_side10_1[h_value] = []
    dataset_side10_2[h_value] = []
    dataset_side10_3[h_value] = []
    print("size = 10, h = " + h_value + ", shape is ")
    print(dataset_side10_augmented[h_value].shape)

# side 12
dataset_side12_1 = {}
dataset_side12_2 = {}
dataset_side12_3 = {}
dataset_side12_augmented = {}

for h_value, data in dataset_side12.items():
    dataset_side12_1[h_value] = rotate1(dataset_side12[h_value], 12)
    dataset_side12_2[h_value] = rotate1(dataset_side12_1[h_value], 12)
    dataset_side12_3[h_value] = rotate1(dataset_side12_2[h_value], 12)
    
    dataset_side12_augmented[h_value] = np.concatenate((dataset_side12[h_value], 
                                                       dataset_side12_1[h_value], 
                                                       dataset_side12_2[h_value], 
                                                    dataset_side12_3[h_value]),
                                                    axis=0)
    dataset_side12_1[h_value] = [] #to not use up so much memory
    dataset_side12_2[h_value] = []
    dataset_side12_3[h_value] = []
    print("size = 12, h = " + h_value + ", shape is ")
    print(dataset_side12_augmented[h_value].shape)

del dataset_side12_1
del dataset_side12_2
del dataset_side12_3
del dataset_side12


# QSL data
#do the same augmentation
dataset_qsl_beta5_1 = {}
dataset_qsl_beta5_2 = {}
dataset_qsl_beta5_3 = {}
dataset_qsl_beta5_augmented = {}

for h_value, data in dataset_qsl_beta5.items():
    dataset_qsl_beta5_1[h_value] = rotate1(dataset_qsl_beta5[h_value], 8)
    dataset_qsl_beta5_2[h_value] = rotate1(dataset_qsl_beta5_1[h_value], 8)
    dataset_qsl_beta5_3[h_value] = rotate1(dataset_qsl_beta5_2[h_value], 8)
    
    dataset_qsl_beta5_augmented[h_value] = np.concatenate((dataset_qsl_beta5[h_value], 
                                                       dataset_qsl_beta5_1[h_value], 
                                                       dataset_qsl_beta5_2[h_value], 
                                                    dataset_qsl_beta5_3[h_value]),
                                                    axis=0)
    dataset_qsl_beta5_1 = {}
    dataset_qsl_beta5_2 = {}
    dataset_qsl_beta5_3 = {}
    print("size = 12, h = " + h_value + ", shape is ")
    print(dataset_qsl_beta5_augmented[h_value].shape)
    

# visualize dataset
for h_value in ['0.05', '0.1', '0.15', '0.2', '0.25','0.3', '0.4', '0.5', '0.6']:
    visualize_data(8, 2, 6, dataset_qsl_beta5, h_value)


    
    
    
    
"""
Set up training and testing sets (from here on we show QSL example only, 8x8 lattice, 64x64 dataset.)
"""
# QSL beta = 5

# #shuffle data to be partitioned into train and test sets
np.random.shuffle(dataset_qsl_beta5_augmented["0.1"])
np.random.shuffle(dataset_qsl_beta5_augmented["0.6"])

#choose training data
x_train_aug = concatenate_X_data([dataset_qsl_beta5_augmented["0.1"], dataset_qsl_beta5_augmented["0.6"]],[dataset_qsl_beta5_augmented["0.1"].shape[0] - 500, dataset_qsl_beta5_augmented["0.6"].shape[0] - 500], [0,0]) #use  all but 500 examples from each
y_train_aug = set_up_labels_Y([dataset_qsl_beta5_augmented["0.1"], dataset_qsl_beta5_augmented["0.6"]],[0,1],[dataset_qsl_beta5_augmented["0.1"].shape[0] - 500, dataset_qsl_beta5_augmented["0.6"].shape[0] - 500])

#testing data (from the same distribution)
x_test_aug = concatenate_X_data([dataset_qsl_beta5_augmented["0.1"], dataset_qsl_beta5_augmented["0.6"]],[500,500],[dataset_qsl_beta5_augmented["0.1"].shape[0] - 500,dataset_qsl_beta5_augmented["0.6"].shape[0] - 500])
y_test_aug = set_up_labels_Y([dataset_qsl_beta5_augmented["0.1"], dataset_qsl_beta5_augmented["0.6"]],[0, 1],[500,500])


print('Data with intermediate h values, will later use NN to makes predictions')

h_testing_qsl5={}
h_testing_qsl5_labels={}

for h, data in dataset_qsl_beta5_augmented.items():
    print('dataset_side8, h = ' + h)
    h_testing_qsl5[h] = concatenate_X_data([data],[data.shape[0]],[0])
    h_testing_qsl5_labels[h] = set_up_labels_Y([data],[1],[data.shape[0]])
    print()
    print()

    
    
    
    
    
    
'''
Logistic regression: define, compile, train, test
'''
def Logistic(input_shape):
    """
    Logistic regression
    """
    X_input = Input(input_shape)
    X = X_input
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=0.00, l2=0.00), name='fc')(X)

    model = Model(inputs = X_input, outputs = X, name='Logistic')
    return model

logistic_model = Logistic((64,64,1))
logistic_model.compile(optimizer = Adam(lr=0.001), loss = "binary_crossentropy", metrics = ["accuracy"])
logistic_model.summary()

# summarize and compile model
logistic_model.fit(x = x_train_aug, y = y_train_aug, epochs = 50, batch_size = 50, shuffle = True)

# make plots of "fraction labelled 1" vs h
plot_this(trend_of_mean(dataset_qsl_beta5_augmented),color='b')
plot_this(evaluate_diff_h_3(logistic_model, h_testing_qsl5, h_testing_qsl5_labels,[k for k in h_testing_qsl5.keys()]))







"""
Convolutional Neural Network: define, compile, train, test
"""
def ConvModel(input_shape):
    """
    Single conv layer.
    """
    X_input = Input(input_shape)
    X = X_input
    X = Conv2D(5, 5, strides = 1, padding='valid', kernel_regularizer=l1_l2(l1=0.00, l2=0.00), name = 'conv0')(X)
    X = Activation('relu')(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=0.0, l2=0.00), name='fc2')(X)

    model = Model(inputs = X_input, outputs = X, name='ConvModel')
    return model

# # one conv layer, sample size = 8
conv_model8 = ConvModel((64,64,1))
print("Total number of parameters:")
print(conv_model8.count_params())
conv_model8.summary()

conv_model8.compile(optimizer = Adam(lr=0.001), loss = "binary_crossentropy", metrics = ["accuracy"])


#train for 50 epochs, printing both training and dev set accuracy in each step; use early stopping
for i in range(50):
    print()
    print()
    print('Epoch: ')
    print(i)
    conv_model8.fit(x = x_train_aug, y = y_train_aug, epochs = 1, batch_size = 50, shuffle = True)
    preds = conv_model8.evaluate(x = x_test_aug, y = y_test_aug)
    print()
    print("Test set accuracy with same distribution:")
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    
# make plots of "fraction labelled 1" vs h    
plot_this(evaluate_diff_h_3(conv_model8, h_testing_qsl5, h_testing_qsl5_labels,[k for k in h_testing_qsl5.keys()]))
   
    