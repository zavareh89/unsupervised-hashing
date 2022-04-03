
import numpy as np
from numpy.linalg import eigh, svd
from scipy.linalg import orth
from scipy.stats import ortho_group

from SH import spectral_hashing
from utilities import normalize_Z


def gram_schmidt_orthogonalization(U, num_vectors=1, random_state=42):
    """
    This function implements gram schmidt orthogonalization method
    to compute num_vectors new vectors that are orthogonal  to the 
    columns of U. The function returns [U, U_hat] that U_hat has 
    num_vectors columns.
    Inputs: 
        U: a (N, dim) matrix 
        num_vectors: how many times this orthogonalization have to execute
    Output:
        concatenation (column-wise) of U and num_vectors columns.  
    """
    def project_v_on_u(v,u):
        """
        This operator projects the vector v orthogonally onto the line spanned by 
        vector u. If u = 0, projection v on 0 is defined 0.
        """
        assert u.shape == v.shape
        assert len(u.shape) == 1
        if np.all(u==0):
            return u
        scalar = np.dot(v,u)/np.dot(u,u)
        return scalar*u

    N, dim = U.shape
    dtype = U.dtype
    R = np.random.RandomState(random_state)
    U_new = np.concatenate([U, np.zeros((N, num_vectors), dtype=dtype)], axis=1)
    U_hat = np.array(R.randn(N, num_vectors), dtype=dtype) # initialization
    for i in range (num_vectors):
        v = U_hat[:, i]
        for j in range(dim):
            v -= project_v_on_u(v, U_new[:, j])
        if np.any(v!=0):
            v /= np.linalg.norm(v)
        U_new[:, dim] = v
        dim += 1    
    return U_new


def DGH(Z, K=16, rho=0.1, TB=300, TG=20, B=None, Y=None, random_state=42, verbose=1):
    """
    This function implements descrete graph hashing method (2014). For more 
    details, see the corresponding paper. 
    Inputs:
        Z: data-to-anchor mapping of train data. shape is (n_samples, n_anchors)
        K: Number of bits for each binary code
        rho:tuning parameter (see equation 3 of the paper)
        TB: iteration numbers of optimizing the B-subproblem
        TG: iteration numbers of optimizing the whole DGH problem
        B: initialization of B matrix. If None, DGH-I initialization is used
        Y: initialization of Y matrix. If None, DGH-I initialization is used
        random_state: used for initialization of gram schmidt orthogonalization
        verbose: 1 or 0. If 1, it prints some information about early convergence
                 and rank of JB matrix in Y-subproblem
    Outputs:
        B: learned binary codes. The values are [-1,1]. shape is (n_samples, K)
        Y: relaxed version of B (with real values). shape is (n_samples, K)
    """
    n_samples = Z.shape[0]
    eps = 1e-4 # controls convergence
    if Y is None:
        Y = spectral_hashing(Z, K=K, thresholding=None)
    if B is None:
        B = np.sign(Y)
    # normalize Z
    Z = normalize_Z(Z)
        
    def B_subproblem(B, Y, L, rho=0.1, TB=300):
        """ 
        signed gradient method (SGM) for B-subproblem (see the algorithm 1 of 
        the paper).
        Note that L = Z.T@B is computed once in the objective function and we 
            brought it here for efficency reasons.        
        """
        for t in range(TB):
            B_conv = B.copy() # used to check convergence
            grad = 2*Z@L + rho*Y
            B = np.sign(np.where(np.abs(grad)>0, grad, B))
            # check convergence
            if np.all(B==B_conv)==True:
                if verbose:
                    print(f"B_subproblem converged early in {t+1} iteration")
                return B
        return B
        
    def Y_subproblem(B, tg, K=8, random_state=42):
        # tg is iteration number starting from 0
        B = B - np.mean(B, axis=0) # equivalent to JB (see paper)
        M = B.T@B
        eig_vals, V = eigh(M)
        eig_vals, V = eig_vals[::-1], V[:, ::-1]
        r = np.argmin((eig_vals/eig_vals[0])>eps) # rank of M 
        if r==0:
            r = K
            if verbose:
                print(f"The rank of JB matrix is {r} (full-rank) in Y"
                      f" subproblem and in iteration {tg+1}")
        else:
            if verbose:
                print(f"The rank of JB matrix is {r} in Y subproblem and" 
                      f" in iteration {tg+1}")
        sigma_inv = np.diag(1/np.sqrt(eig_vals[:r]))
        U = B@V[:,:r]@sigma_inv
        if r==K: # full-rank case
            return np.sqrt(n_samples)*(U@V.T)        
        # If the M is not full rank, we must generate U_hat
        U = np.concatenate([U, np.ones((n_samples,1), dtype=U.dtype)], axis=1)
        U_new = gram_schmidt_orthogonalization(U, num_vectors=K-r,
                                               random_state=random_state)
        idx = np.setdiff1d(np.arange(K+1), r)
        return np.sqrt(n_samples)*(U_new[:, idx]@V.T)
    
    def objective_func(B, Y, rho=0.1):
        L = Z.T@B
        return np.trace(L.T@L) + rho*np.trace(B.T@Y), L
    
    Q_new, L = objective_func(B, Y, rho=rho)
    for tg in range(TG):
        Q_old = Q_new
        # B subproblem
        B = B_subproblem(B, Y, L, rho=rho, TB=TB)
        # Y subproblem
        Y = Y_subproblem(B, tg, K=K, random_state=random_state)
        # check convergence
        Q_new, L = objective_func(B, Y, rho=rho)
        if ((Q_new-Q_old)/Q_old)<eps:
            if verbose:
                print(f"The DGH problem converged early in {tg+1} iterations")
            return B, Y   
    return B, Y


def SHSR(Z, K=16, n_iter=20, Q=None, random_state=42, verbose=1):
    """
    This function implements Spectral Hashing with Spectral Rotation (SHSR) method.
    For more details, see "Discrete Spectral Hashing for Efficient Similarity 
    Retrieval" paper (section III). 
    Inputs:
        Z: data-to-anchor mapping of train data. shape is (n_samples, n_anchors)
        K: Number of bits for each binary code
        n_iter: maximum number of iterations.
        Q: initialization of Q matrix. If None, A random orthogonal matrix is used
        random_state: used for initialization of Q (if it is None)
        verbose: 1 or 0. If 1, it prints some information about early convergence
    Outputs:
        B: learned binary codes. The values are [-1,1]. shape is (n_samples, K)
        F: output of spectral hashing method. shape is (n_samples, K)
        Q: learned orthogonal matrix. shape is (K, K)
    """
    n_samples = Z.shape[0]
    # apply spectral hashing 
    F = spectral_hashing(Z, K=K, thresholding=None)
    Error = 1000 # used for convergence check
        
    def B_subproblem(F, Q):
        M = F@Q
        half =  n_samples//2
        Idx = np.argsort(-M, axis=0)
        B = np.zeros((n_samples, K), dtype=np.float32)
        for i in range(K):
            B[Idx[:half, i], i] = 1.
            B[Idx[half:, i], i] = -1.
        return B  
    
    def Q_subproblem(F, B):
        G = B.T@F
        U, diag_mat, V_transpose = svd(G)
        V = V_transpose.T
        Q = V@U.T
        return Q
    
    # initialization of Q
    if Q is None:
        Q = ortho_group.rvs(K, random_state=random_state)
    assert Q.shape == (K, K)
    B = B_subproblem(F, Q)
    
    for it in range(n_iter):                
        B_old = B                                         
        Q = Q_subproblem(F, B)
        B = B_subproblem(F, Q)
        # convergence check
        Error = np.mean(np.abs(B-B_old))
        if verbose>0:
            print(f"Mean absolute error is {Error} after {it+1} iterations")
        if Error < 1e-4:
            if verbose>0:
                print(f"\nThe SHSR problem converged early in {it+1} iterations")
            return B, F, Q
    return B, F, Q


def DSH(Z, K=16, alpha=0.1, F=None, n_iter=30, Ng_iter=30, random_state=42, verbose=1):
    """
    This function implements DISCRETE SPECTRAL HASHING (DSH) method.
    For more details, see "Discrete Spectral Hashing for Efficient Similarity 
    Retrieval" paper (section V). 
    Inputs:
        Z: data-to-anchor mapping of train data. shape is (n_samples, n_anchors)
        K: Number of bits for each binary code
        alpha: regularization term for quantization loss
        F: initialization of F matrix. If None, A random orthogonal matrix is used
        n_iter: maximum number of iterations.
        Ng_iter: maximum number of iterations of Generalized Power method
        random_state: used for initialization of F
        verbose: 1 or 0. If 1, it prints some information about early convergence
    Outputs:
        B: learned binary codes. The values are [-1,1]. shape is (n_samples, K)
    """
    n_samples = Z.shape[0]
    # normalize Z
    Z = normalize_Z(Z)
        
    def F_subproblem(Z, B, F, Ng_iter):
        # this subprolem is solved using Generalized Power Iteration (GPI)4
        for it_g in range(Ng_iter):
            F_old = F
            M = 2*(Z@(Z.T@F)) + 2*alpha*B
            U, diag_mat, V_transpose = svd(M, full_matrices=False)
            F = U@(np.sqrt(n_samples)*V_transpose)
            Error = np.mean(np.abs(F-F_old))
            if Error < 1e-4:
                if verbose>0:
                    print("early convergence of Generalized Power Iteration "\
                              f"algorithm in {it_g+1} iteration")
                return F
        return F

    def B_subproblem(F):
        half =  n_samples//2
        Idx = np.argsort(-F, axis=0)
        B = np.zeros((n_samples, K), dtype=np.float32)
        for i in range(K):
            B[Idx[:half, i], i] = 1.
            B[Idx[half:, i], i] = -1.
        return B     
    
    # F initialization
    if F is None:
        R = np.random.RandomState(random_state)
        F = np.float32(orth(R.rand(n_samples, K)))
        assert F.shape==(n_samples, K)
    #  initialization
    B = B_subproblem(F)
    # main loop
    for it in range(n_iter):
        if verbose>0:
            print(f"iteration {it+1}:")      
        B_old = B
        F = F_subproblem(Z, B, F, Ng_iter)
        B = B_subproblem(F)
        # convergence check
        Error = np.mean(np.abs(B-B_old))
        if verbose>0:
            print(f"Mean absolute error is {Error}")            
        if Error < 1e-4:
            if verbose>0:
                print(f"\nThe DSH problem converged early in {it+1} iterations")
            return B          
    return B
