
import numpy as np
from numpy.linalg import eigh
from sklearn.preprocessing import StandardScaler

from Datasets import load_dataset
from evaluate import precision_recall, precision_radius, mAP, Macro_AP, return_all_metrics
from ITQ import ITQ

#%% parameter initialization
path = r'cifar10_gist512' # folder containing dataset
dataset_name = 'cifar10_gist512'
K = 16 # number of bits
random_state = 42

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

#%% compute M matrix and its eigen-values (PCA step)
M = (train_features.T@train_features)/n_train
eig_values, W = eigh(M)
idx = np.argsort(-eig_values)
eig_values, W = eig_values[idx], W[:,idx]
Wx = W[:,:K]

#%% Learning rotation matrix
V = train_features@Wx
R, Q_loss = ITQ(V, n_iter=50, random_state=random_state)

#%% compute binary codes for train and test features
B = np.sign(V@R)
test_codes = np.sign(test_features@Wx@R)

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
