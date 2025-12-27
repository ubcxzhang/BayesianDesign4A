"""
mvn.py

A GPU-accelerated 2D Multivariate Normal PDF and CDF implementation in CuPy.

References
----------
Alan Genz, Numerical Computation of Multivariate Normal Probabilities,
Journal of Computational and Graphical Statistics, 
Jun 1992, Vol. 1, No. 2, pp. 141-149
"""

import cupy as cp
from cupyx.scipy import special
from . import rndgenerator


def van_der_corput(n: int, base=2, start_index=0, scramble=False, seed=None):
    """
    A van der Corput sequence generator for use in the Quasi Monte Carlo lattice method.

    Parameters
    ----------
    n : int
        Length of generated sequence
    base : int, optional
        (default is 2)
    start_index : int, optional
        What index to start generating at
        (default is 0)
    scramble : bool, optional
        Whether the order of the returned sequence should be randomized
        (default is False)
    seed : int, optional
        (default is None)
    """
    rndgenerator.seed(seed)
    rs = rndgenerator.get_random_state()
    sequence = cp.zeros(n)

    quotient = cp.arange(start_index, start_index + n)
    b2r = 1 / base
    while (1 - b2r) < 1:
        remainder = quotient % base
        if scramble:
            # permutation must be the same for all points of the sequence
            perm = rs.permutation(base)
            remainder = perm[cp.array(remainder).astype(int)]
        sequence += remainder * b2r
        b2r /= base
        quotient = (quotient - remainder) / base
    return sequence

def pdf1D(mu, Sigma, x):
    """
    Returns the value of the PDF for a 1D normal distribution at x.

    Parameters
    ----------
    mu : float
        The mean of the normal distribution.
    Sigma : float
        The variance of the normal distribution (sigma^2).
    x : float
        Point in the function to be measured.
    """
    # Compute the deviation from the mean
    x_mu = x - mu

    # Compute the PDF of the 1D normal distribution
    return (1.0 / cp.sqrt(2 * cp.pi * Sigma)) * cp.exp(-0.5 * (x_mu ** 2) / Sigma)

def pdf2D(mu, Sigma, x):
    """
    Returns the value of the PDF for a 2D multivariate normal distribution at x.

    Parameters
    ----------
    mu : cupy.ndarray(2)
        The mean of the normal distribution
    Sigma : cupy.ndarray(2,2)
        The covariance matrix of the normal distribution
    x : cupy.ndarray(2)
        Point in the function to be measured
    """
    det_Sigma = Sigma[0,0]*Sigma[1,1]-Sigma[0,1]**2
    inv_Sigma = -Sigma.copy()                                   # inv_Sigma = -Sigma  
    inv_Sigma[0,0] = Sigma[1,1]
    inv_Sigma[1,1] = Sigma[0,0]
    inv_Sigma = inv_Sigma/det_Sigma
    x_mu = x-mu
    return 0.5*cp.exp(-0.5*x_mu.T@inv_Sigma@x_mu)/(cp.pi*cp.sqrt(cp.abs(det_Sigma)))

# @cp.fuse
def cdf1D(x,mean=0.0,std=1):
    """
    Returns the CDF for a 1D normal distribution at x  - a single real number.

    Parameters
    ----------
    x : cupy.ndarray(2)
        Point in the function measured
    mean : float
        The mean of the normal distribution (default is 0)
    std : 1
        The standard deviation of the normal distribution
    """
    return 0.5*(1. + special.erf((x-mean)*cp.reciprocal(std)/cp.sqrt(2.)))

@cp.fuse
def icdf1D(x,mean=0.0,std=1.):
    """
    Returns the inverse CDF for a 2D multivariate normal distribution at x.

    Parameters
    ----------
    x : cupy.ndarray(2)
        Point in the function measured
    mean : float
        The mean of the normal distribution (default is 0)
    std : 1
        The standard deviation of the normal distribution
    """
    return mean + std * special.erfinv(2. * x - 1.) * cp.sqrt(2.)

def cdf2DLattice(sigma=cp.eye(2),ub=cp.array([+cp.inf,+cp.inf]), 
        Nmax=50000, lattice_points=None):
    """
    Returns a multivariate MVN CDF at a single point
    
    References
    ----------
    Alan Genz algorithm customized for 2D case
    Journal of Computational and Graphical Statistics, 
    Jun 1992, Vol. 1, No. 2, pp. 141-149

    Parameters
    ----------
    sigma : cupy.ndarray(2,2)
        The covariance matrix of the normal distribution
    ub : cupy.ndarray(2)
        "Upper Bound". Point in the function measured 
        (default is [cupy.inf, cupy.inf])
    Nmax: int
        (default is 50000)
    lattice_points: cupy.ndarray(Nmax)
        vector of lattice points used for QMC method
    """
    # Cholesky decomposition of sigma:
    C = cp.zeros((2,2))
    
    fsigma = sigma.copy()
    b = ub.copy()
    
    # Flip the bounds if the inner integration interval is small
    if b[0] > b[1]:
        fsigma[0,0] = sigma[1,1]
        fsigma[1,1] = sigma[0,0]
        b[0] = ub[1]
        b[1] = ub[0]
    C[0,0] = cp.sqrt(fsigma[0, 0]); C[1,0]=fsigma[1, 0]/cp.sqrt(fsigma[0, 0]);
    C[1,1] = cp.sqrt(fsigma[1, 1]-fsigma[1, 0]**2/fsigma[0, 0])
    e0 = 1.
    if cp.isfinite(b[0]):
        e0 = cdf1D(b[0]/C[0,0])
    
    y0vec = icdf1D(lattice_points*e0)
    e1vec = cp.ones(Nmax)
    if cp.isfinite(b[1]):
        e1vec = cdf1D((b[1]-C[1,0]*y0vec)/C[1,1])
    return e0 * e1vec.mean()

def redVecCDF2DLattice(sigma=cp.eye(2),ub=cp.array([[+cp.inf,+cp.inf]]), 
                       Nmax=5000, slice_size=5000,lattice_points=None):
    """
    Vectorized multivariate MVN CDF calculation.

    Parameters
    ----------
    sigma : cupy.ndarray(2,2)
        The covariance matrix of the normal distribution
    ub : cupy.ndarray(N_DeltaXt, 2)
        "Upper Bound". An array of points in the function to measure and average. 
        (default is [N_DeltaXt][cupy.inf, cupy.inf])
    Nmax: int
        (default is 5000)
    slice_size: int
        (default is 5000)
    lattice_points: cupy.ndarray(Nmax)
        vector of lattice points used for QMC method
    """
    # Cholesky decomposition of sigma:
    C = cp.zeros((2,2))
    C[0,0] = cp.sqrt(sigma[0, 0]); C[1,0]=sigma[1, 0]/cp.sqrt(sigma[0, 0]);
    C[1,1] = cp.sqrt(sigma[1, 1]-sigma[1, 0]**2/sigma[0, 0])
    # print(C)
    #########################Not a correctness bug, but better practice:####################
    # e0 = cp.ndarray(slice_size)  # the inf ones are set to 1
    # b = cp.ndarray(slice_size)
    # e1vec = cp.ndarray((Nmax,slice_size))
    
    e0 = cp.empty(slice_size)
    b  = cp.empty(slice_size)
    e1vec = cp.empty((Nmax, slice_size))
    ########################################################################################
    
    n_pts = ub.shape[0]//slice_size*slice_size
    res = cp.zeros(n_pts) # final results - only the multiple slices of points is considered
    for sl in range(0,n_pts,slice_size):
        slend = sl+slice_size # slice end idx
        b = ub[sl:slend,0]
        e0 = cdf1D(b/C[0,0])   
        y0vec = icdf1D(lattice_points*e0) #icdf1D(d0+lattice_points*f0)
        b = ub[sl:slend,1]
        e1vec = cdf1D((b-C[1,0]*y0vec)/C[1,1])
        res[sl:slend] = e0*e1vec.mean(axis=0) #f0y
        # print(e0,y0vec,e1vec)
    return res.mean()

# if __name__ == "__main__":
#     random_seed=101
#     rndgenerator.seed(random_seed)
#     Nmax=5000
#     lattice_points = van_der_corput(n=Nmax, scramble=True, seed=random_seed)
#     #lattice_points_col = lattice_points.reshape(-1, 1)
#     #print(lattice_points)
#     ### Evaluation at many points - for a single point check the next call instead.
#     print(redVecCDF2DLattice(ub=cp.array([[0,0]]),sigma=cp.ones((2,2)),lattice_points=lattice_points,slice_size=1))
#     print(cdf2DLattice(sigma=cp.ones((2,2)),Nmax=Nmax,ub=cp.array([+cp.inf,+cp.inf]),lattice_points=lattice_points))
#     print(cdf2DLattice(sigma=cp.ones((2,2)),Nmax=Nmax,ub=cp.array([0,0]),lattice_points=lattice_points))