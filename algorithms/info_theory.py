import numpy as np
from numpy import log2

eps = 1e-40


def _normalize(P):
    """ normalize probability distribution """
    s = sum(P)
    if s == 0.:
        raise ValueError("input distribution has sum zero")
    return P / s


def blahut_arimoto(P_yx, q_x, epsilon = 0.001, deterministic = False, iters = 20):
    """
    Compute the channel capacity C of a channel p(y|x) using the Blahut-Arimoto algorithm. To do
    this, finds the input distribution p(x) that maximises the mutual information I(X;Y)
    determined by p(y|x) and p(x).

    P_yx : defines the channel p(y|x)
    iters : number of iterations
    """
    P_yx = P_yx + eps
    if not deterministic:
        T = 1
        i = 0
        while T > epsilon and i < iters:
            # update PHI
            PHI_yx = (P_yx*q_x.reshape(1,-1))/(P_yx @ q_x).reshape(-1,1)
            r_x = np.exp(np.sum(P_yx*log2(PHI_yx), axis=0))
            # channel capactiy
            C = log2(np.sum(r_x))
            # check convergence
            T = np.max(log2(r_x/q_x)) - C
            # update q
            q_x[:] = _normalize(r_x + eps)
            i+=1
        if C < 0:
            C = 0
        return C
    else:
        # assume all columns in channel matrix are peaked on a single state
        # log of number of reachable states
        return log2(np.sum(P_yx.sum(axis=1) > 0.999))
