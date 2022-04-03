
"""
This code implements out-of-sample extension technique in hashing methods.
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from utilities import to_Z

def projection_matrix(B, Z):
    """
    Projectetion matrix maps anchor-based represntation of samples to binary
    codes.
    Inputs:
        B: binary codes or approximated binary codes (with real values) of
           training samples. If B is binary, values must be [-1,1].
           shape of B is (n_samples, K) which K is number of bits.
        Z: anchor-based represntation of training samples. shape is 
           (n_samples, n_anchors).
    Output:
        Projection matrix. Shape is (K, n_anchors).
    """

    D = np.array(1/(np.sum(Z,axis=0)), dtype='float32') 
    D = np.diag(D.flatten())
    
    return B.T@Z@D


def out_of_sample_binary_codes(X_test, B, Z_train, anchors, Z_spec):
    """
    This function generates binary code for new (unseen) samples.
    Ref: Discrete Graph Hashing paper 2014. 
    Inputs:
        X_test: new samples. shape is (n_test, n_features).
        B: learned binary codes or approximated binary codes (with real values)
           of training samples. If B is binary, values must be [-1,1].
           shape of B is (n_train_samples, K) which K is number of bits.
        Z_train: anchor-based represntation of training samples. shape is 
           (n_train_samples, n_anchors).
        anchors: anchor points. shape is (n_anchors, n_features).
        Z_spec: a dictionary that specifies how to compute Z. Keys should be
                'n_anchors', 's', 'sigma', and 'metric'.
    Output:
        B_test: new binary codes for test samples. shape is (n_test, K), which 
                K is the number of bits.
    """
    
    W = projection_matrix(B, Z_train)
    Z = to_Z(X_test, anchors, Z_spec)
    B_test = Z@W.T
    return np.sign(B_test)


def KSH_binary_codes(X_test, Zp, A, b, metric='linear' ):
    """
    This function generates binary code for new (unseen) samples using KSH method.
    Ref: Scalable Similarity Search with Optimized Kernel Hashing paper 2010. 
    Inputs:
        X_test: new samples. shape is (n_test, n_features).
        Zp: landmark samples. shape is (P, n_features).
        A: combination weights. shape is (P, K) which K is the number of bits.
        b: combination weights (biases). shape is (K,1).
        metric: string, or callable.The metric to use when calculating kernel between
            instances in a feature array.If metric is a string, it must be one of the
            metrics in sklearn.pairwise.PAIRWISE_KERNEL_FUNCTIONS. If metric is a 
            callable function, it is called on each pair of instances (rows) and the
            resulting value recorded. The callable should take two vectors as input 
            and return the corresponding kernel value as a single number.
        
    Output:
        B_test: new binary codes for test samples. shape is (n_test, K), which 
                K is the number of bits.
    """
    
    kernels = pairwise_kernels(Zp, X_test, metric=metric) 
    B = np.sign(A.T@kernels - b)

    return np.transpose(B)