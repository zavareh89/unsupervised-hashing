
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

from Affinity_matrix import Affinity
from Datasets import load_dataset
from Discrete_hashing_methods import DGH
from evaluate import precision_recall, precision_radius, mAP, Macro_AP, return_all_metrics
from out_of_sample import out_of_sample_binary_codes

#%% parameter initialization
method_name = 'DGH' 
path = r'nuswide_vgg' # folder containing dataset
dataset_name = 'nuswide_vgg'
K = 16 # number of bits

## The following parameters are related to Affinity matrix
n_anchors = 300 # Number of anchors to compute (sparse) affinity matrix
s = 3 # Number of selection for nearest anchors
dist_metric = 'euclidean' # Distance metric used for computing affinity matrix
anchor_init = 'kmeans' # The method used to compute anchors.
random_state = 42
sigma = None # bandwitch to normalize distances (see to_Z function)
Z_spec = {'n_anchors':n_anchors, 's':s, 'sigma':sigma, 'metric':dist_metric}

#%% load data
train_features, train_labels, test_features, test_labels = load_dataset(
    dataset_name, path=path, one_hot=False)
train_features = train_features.astype('float32')
test_features = test_features.astype('float32')
n_train = train_features.shape[0]

#%% normalization
scaler = StandardScaler(with_mean=True, with_std=True)
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

#%% compute affinity matrix (Z and achor matrices)
Z, anchors = Affinity(train_features, Z_spec, random_state=random_state,
                      init=anchor_init)
# # save Z, and anchors
# np.savez(os.path.join('saved_arrays', 'Z_and_anchors_' + dataset_name + '.npz'),
#          Z=Z, anchors=anchors)

# # load Z, and anchors
# dict_arr = np.load(os.path.join('saved_arrays', 'Z_and_anchors_' + dataset_name + '.npz'))
# Z, anchors = dict_arr['Z'], dict_arr['anchors']

#%% Learning binary codes
B, Y = DGH(Z, K=K, rho=0.1, TB=300, TG=20, random_state=random_state, verbose=1)

#%% Generate binary codes using out of sample technique
test_codes = out_of_sample_binary_codes(test_features, B, Z, anchors, Z_spec)

#%% evaluation
if dataset_name=='labelme_vggfc7' :
     macro_AP = Macro_AP(B>0, train_labels, test_codes>0, test_labels, 
                         num_return_NN=None)
     print(f"Macro average precision for K={K} is {macro_AP}")
else:
     ## precision and recall are computed @ M_set points
    M_set = np.arange(250, n_train, 250) 
    MAP, precision, recall, precision_Radius = return_all_metrics(
        B>0, train_labels, test_codes>0, test_labels, M_set, Radius=2)
    print(f"K={K}")
    print(f"MAP={MAP}")
    print(f"precision_Radius={precision_Radius}")
    if dataset_name=='nuswide_vgg':
        print(f"precision_5000={precision[19]}")
    else:
        print(f"precision_1000={precision[3]}")
