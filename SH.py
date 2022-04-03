
import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.metrics.pairwise import pairwise_kernels

from utilities import normalize_Z


def spectral_hashing(Z, K=8, thresholding='sign'):
    """ Spectral hashing
    Z: data-to-anchor mapping of train data. shape is (n_samples, n_anchors)
    K: Number of bits for each binary code
    thresholding: The thresholding method used to convert relaxed (continuous) 
                  codes to binary codes. If None, no thresholding is applied.
    
    Output: 
        B: The learned binary codes. It has shape (n_samples,K)
    """    
    # construct M matrix (See AGH paper)
    n_train = Z.shape[0]
    Z = csr_matrix(Z, dtype='float32') 
    D = np.array(1/np.sqrt(np.sum(Z,axis=0)), dtype='float32')
    D = diags(D.flatten(), dtype='float32')
    M = (D@Z.T@Z@D).toarray()
    # eigenvalue decomposition of M
    lambdas, eigen_vecs = np.linalg.eigh(M)
    idx_eig = np.argsort(-lambdas)
    lambdas, eigen_vecs = lambdas[idx_eig[1:K+1]], eigen_vecs[:, idx_eig[1:K+1]]
    
    # compute Affinity eigenvectors
    S = np.diag(1/np.sqrt(lambdas))
    B = Z@D@eigen_vecs@S
    B = np.sqrt(n_train)*B

    if thresholding is None:
        return B
    
    # compute binary codes
    B = np.sign(B)
    
    return B


def kernel_spectral_hashing(train_data, Z, Zp, K=16, metric='linear',
                            regularizer_term=0, thresholding='sign'):
    """ Kernel Spectral hashing (KSH) method
    Ref: Scalable Similarity Search with Optimized Kernel Hashing paper 2010.
    train_data: shape is (n_samples,n_features)
    Z: data-to-anchor mapping of train data. shape is (n_samples, n_anchors)
    Zp: landmark samples. shape is (P, n_features).
    K: Number of bits for each binary code
    metric: string, or callable.The metric to use when calculating kernel between
        instances in a feature array.If metric is a string, it must be one of the
        metrics in sklearn.pairwise.PAIRWISE_KERNEL_FUNCTIONS. If metric is a 
        callable function, it is called on each pair of instances (rows) and the
        resulting value recorded. The callable should take two vectors as input 
        and return the corresponding kernel value as a single number.
    regularizer_term: the scalar term used in KSH objective function (see eq 2)
    thresholding: The thresholding method used to convert relaxed (continuous) 
                  codes to binary codes. If None, no thresholding is applied.
    
    Output: 
        B: The learned binary codes. It has shape (n_samples,K)
        A: combination weights. shape is (P, K) which K is the number of bits.
        b: combination weights (biases). shape is (K,1).
    """    
    # normalize Z
    Z = normalize_Z(Z)
    
    N = train_data.shape[0] # number of samples
    K_PN = pairwise_kernels(Zp, train_data, metric=metric)
    k_avg = np.mean(K_PN, axis=1, keepdims=True)
    K_PP = 0 
    if regularizer_term != 0:
        K_PP = pairwise_kernels(Zp, Zp, metric=metric)
    # note that in low-rank representation of affinity matrix, D=I
    #D = affinity_matrix.sum(axis=0).A1 + affinity_matrix.sum(axis=1).A1
    # A1 method flattens numpy.matrix (flatten method does not work!)
    #D = diags(D/2)
    C = K_PN@(K_PN.T - Z@(Z.T@K_PN.T)) + regularizer_term*K_PP
    G = (K_PN@K_PN.T)/N + k_avg@k_avg.T
    
    # we use eigenvalue decomposition instead of SVD, because G matrix is PSD.
    Lambda0, T0 = np.linalg.eig(G)
    Lambda0 = np.abs(Lambda0)
    idx_eig = np.argsort(-Lambda0)
    Lambda0, T0 = Lambda0[idx_eig], T0[:, idx_eig]
    Lambda, T = Lambda0[:K], T0[:,:K]
    
    # compute A_hat and A
    Lambda_half_inv = np.diag(1/np.sqrt(Lambda))  
    C_hat = Lambda_half_inv@T.T@((C+C.T)/2)@T@Lambda_half_inv
    eig_temp, A_hat = np.linalg.eig(C_hat)
    idx_eig = np.argsort(eig_temp)
    A_hat = A_hat[:, idx_eig]
    A = T@Lambda_half_inv@A_hat
    b = A.T@k_avg
    
    # compute hash codes for training samples
    B = A.T@K_PN - b

    if thresholding is None:
        return np.transpose(B), A, b
    
    # compute binary codes
    B = np.sign(B)
    
    return np.transpose(B), A, b