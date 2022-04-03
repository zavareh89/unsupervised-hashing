
# ITQ function (see ITQ paper)
import numpy as np
from numpy.linalg import svd, norm
from scipy.stats import ortho_group

def ITQ(V, n_iter=50, random_state=None):
    c = V.shape[1] # number of dimensions or bits
    R0 = ortho_group.rvs(c, random_state=random_state)
    Q_loss = np.zeros((n_iter,)) # quantization loss
    
    for it in range(n_iter):
        if it==0:
            VR=V@R0
            
        # compute quantization loss
        if it>0:
            VR = V@R
            # note that the following is Q_loss of previous iteration
            Q_loss[it-1] = (norm(B-VR))**2
            
        # fix R and update B
        B = np.sign(VR)
        
        # fix B and update R
        S, omega, Shat_transpose = svd(B.T@V)
        Shat = Shat_transpose.T
        R = Shat@S.T
        
        # compute the last quantization loss
        if it==(n_iter-1):
            VR = V@R
            Q_loss[it] = (norm(B-VR))**2
    
    return R,Q_loss/(c**2)