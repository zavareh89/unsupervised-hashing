
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras

from Datasets import load_dataset
from evaluate import precision_recall, precision_radius, mAP, Macro_AP, return_all_metrics
from ReSH import create_dense_network, train_ReSH

#%% parameter initialization
method_name = 'ReSH'
path = r'cifar10_gist512' # folder containing dataset
dataset_name = 'cifar10_gist512'
K = 16 # number of bits
n_epochs = 70 # number of epochs for model training
r = 0.05 # the fraction of total train examples which is randomly chosen ...
    # ... in each epoch to approximate the loss function.
# see train_ReSH documentation for more details about Rprop_params keys.
Rprop_params = {'delta_0':0.07, 'delta_min':1e-6, 'delta_max':50.,
                'scale_down':0.5, 'scale_up':1.2}

#%% load data
train_features, train_labels, test_features, test_labels = load_dataset(
    dataset_name, path=path, one_hot=False)
train_features = train_features.astype('float32')
test_features = test_features.astype('float32')
n_train, n_features = train_features.shape

#%% normalization
#scaler = StandardScaler(with_mean=True, with_std=True)
scaler = MinMaxScaler(feature_range=(-1, 1))
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

#%% create neural net model
if n_features <= 1024:
    hidden_dim = np.int32(np.power(2, np.floor(np.log2(n_features))))
else:
    hidden_dim = 1024
model = create_dense_network(input_dim=n_features, hidden_dim=hidden_dim,
                             output_dim=K)

#%% train the model using ReSH and Rprop optimization
model, cost_values, train_outputs = train_ReSH(train_features, model, K=K,
                                               n_epochs=n_epochs, r=r, p=8,
                                               Rprop_params=Rprop_params)

#%% constructing train and test binary codes
B = train_outputs>0.5
test_outputs = model.predict(test_features, batch_size=512)
test_codes = test_outputs>0

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

