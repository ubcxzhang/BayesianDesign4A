import cupy as cp
import numpy as np
import time
from scipy.stats import norm # for testing

# from mvn import van_der_corput, cdf1D, redVecCDF2DLattice, cdf2DLattice
# from BayesSampler import BayesSampler
# from Optimizer import ZSQP, ZSQP_1d, PDFO, PDFO_1d

from .mvn import van_der_corput, cdf1D, redVecCDF2DLattice, cdf2DLattice
from .BayesSampler import BayesSampler
from .Optimizer import ZSQP, ZSQP_1d


def power(Z, Xt, t=0.5, r=0.8, Is=211.0,
          Sigma_0=None, Delta_Xt=None,
          lattice_points_col=None,
          Nmax=6000, slice_size=10000):
    """
    Returns a computed power based on the power function Q(Z,Xt)

    Parameters
    ----------
    Z : cupy.ndarray(2)
        The threshold values of the entire population and the subgroup under consideration
    Xt : cupy.ndarray(2)
        A single sample from posterior distribution Xₜ
    t : float
        'time', the point in the trial the test takes place
        (default is 0.5)
    r : float
        Subgroup proportion vs whole
        (default is 0.9)
    Is : float
        Total number of information units
        (default is 211)
    Sigma_0 : cupy.ndarray(2,2)
        Covariance matrix of multivariate normal distribution 
    Delta_Xt : cupy.ndarray(N_DeltaXt, 2)
        Vector of samples from posterior distribution Xₜ
    lattice_points_col : cupy.ndarray(Nmax, 2)
        Vector of lattice points in columnar form
    Nmax : int
        Number of lattice points
    slice_size : int
        Maximum size of vector to handle concurrently
        This parameter can be raised or lowered based on GPU performance and memory capacity
        (default is 10000)
    """
    sqrt1mt = cp.sqrt(1 - t)
    ZXt = (Z - cp.sqrt(t) * Xt) / sqrt1mt
    Deltat = Delta_Xt * cp.array([sqrt1mt * cp.sqrt(Is),
                                  sqrt1mt * cp.sqrt(r * Is)])
    res = redVecCDF2DLattice(sigma=Sigma_0, ub=ZXt - Deltat,
                             Nmax=Nmax, lattice_points=lattice_points_col, slice_size=slice_size)
    return res

def power_t(Z, Xt, t=0.5, r=0.8, Is=211.0,
          Sigma_0=None, Delta_true=None,
          lattice_points=None,
          Nmax=6000):
    """
    Returns a computed power based on the power function Q(Z,Xt)

    Parameters
    ----------
    Z : cupy.ndarray(2)
        The threshold values of the entire population and the subgroup under consideration
    Xt : cupy.ndarray(2)
        A single sample from posterior distribution Xₜ
    t : float
        'time', the point in the trial the test takes place
        (default is 0.5)
    r : float
        Subgroup proportion vs whole
        (default is 0.9)
    Is : float
        Total number of information units
        (default is 211)
    Sigma_0 : cupy.ndarray(2,2)
        Covariance matrix of multivariate normal distribution
    Delta_Xt : cupy.ndarray(N_DeltaXt, 2)
        Vector of samples from posterior distribution Xₜ
    lattice_points_col : cupy.ndarray(Nmax, 2)
        Vector of lattice points in columnar form
    Nmax : int
        Number of lattice points
    slice_size : int
        Maximum size of vector to handle concurrently
        This parameter can be raised or lowered based on GPU performance and memory capacity
        (default is 10000)
    """
    sqrt1mt = cp.sqrt(1 - t)
    ZXt = (Z - cp.sqrt(t) * Xt) / sqrt1mt
    Deltat = Delta_true * cp.array([sqrt1mt * cp.sqrt(Is),
                                  sqrt1mt * cp.sqrt(r * Is)])
    res = cdf2DLattice(sigma=Sigma_0, ub=ZXt - Deltat,
                       Nmax=Nmax, lattice_points=lattice_points)
    return res
# Express the constraint
def constraint(Z, Xt, t=0.5, Sigma_0=None,
               lattice_points=None,
               Nmax=6000, alpha=0.025):
    """
    Returns a computed power based on the constraint function alpha(Z,Xt)

    Parameters
    ----------
    Z : cupy.ndarray(2)
        The threshold values of the entire population and the subgroup under consideration
    Xt : cupy.ndarray(2)
        A single sample from posterior distribution Xₜ
    t : float
        'time', the point in the trial the test takes place
        (default is 0.5)
    Sigma_0 : cupy.ndarray(2,2)
        Covariance matrix of multivariate normal distributio
    lattice_points : cupy.ndarray(Nmax, 2)
        Vector of lattice points
    alpha : float, optional
        The target Type I error
        (default is 0.025)

    """
    ZXt = (Z - cp.sqrt(t) * Xt) / cp.sqrt(1 - t)
    res = cdf2DLattice(sigma=Sigma_0, ub=ZXt,
                       Nmax=Nmax, lattice_points=lattice_points)
    return -1. + alpha + res  # The constraint is now constraint()>=0


def get_optimalPower(t=0.5, r=0.8, Is=550.0, p_1=0.75,
                     delta=0.36, d=0.24, random_seed=101,
                     Sigma_1=None, Sigma1_coeff=0.01,
                     Xt=cp.array([2.1, 2.2]), alpha=0.025,
                     Nmax=6000, N_DeltaXt=6000, slice_size=20000,
                     initial_Z=np.array([1., 2.]),
                     solver=ZSQP, verbose=False):
    """
    Returns an OptimizeResult for the maximization problem 
            Q(Xt,Z) s.t. alpha(Xt,Z) == alpha,
    thus controlling for Type I error.
    """
    # Generate lattice points
    lattice_points = van_der_corput(Nmax, scramble=True, seed=random_seed)
    lattice_points_col = lattice_points.reshape(-1, 1)

    bsampler = BayesSampler(t=t, r=r, Is=Is, p_1=p_1,
                            d=d, delta=delta, Sigma_1=Sigma_1, Sigma1_coeff=Sigma1_coeff,
                            random_seed=random_seed)
    Sigma_0 = bsampler.Sigma_0
    Delta_Xt = bsampler.sample_Delta_posterior(Xt=Xt, N=N_DeltaXt)

    # Solve using minimization solver
    start_time = time.time()
    sqp_minimizer = solver(
        power=lambda Z: np.array([
            power(Z=cp.array(Z), Xt=Xt, t=t, r=r, Is=Is,
                  Sigma_0=Sigma_0, Delta_Xt=Delta_Xt,
                  Nmax=Nmax, lattice_points_col=lattice_points_col,
                  slice_size=slice_size).get()]),
        constraint=lambda Z: np.array([
            constraint(Z=cp.array(Z), Xt=Xt, t=t,         # <-- remove r=r
                       alpha=alpha, Sigma_0=Sigma_0,
                       Nmax=Nmax, lattice_points=lattice_points).get()]))
    res = sqp_minimizer.minimize(x0=initial_Z, verbose=verbose)
    end_time = time.time()

    tabular_results = {
        'Z': f"({res['x'][0]}, {res['x'][1]})",
        'power': 1 - res['fun'],
        'alpha': alpha - constraint(cp.array(res['x']), Xt=Xt, t=t,
                                    Sigma_0=Sigma_0, lattice_points=lattice_points,
                                    Nmax=Nmax, alpha=alpha),
        'runtime': round(end_time - start_time, 6), # 'runtime': round(end_time - start_time, digits=6),
    }

    return tabular_results


def get_PowerZ_bsampler(bsampler=BayesSampler(), random_seed=101,
                        Xt=cp.array([2.1, 2.2]), alpha=0.025,
                        Nmax=6000, N_DeltaXt=6000, slice_size=20000,
                        initial_Z=np.array([1., 2.]),
                        solver=ZSQP, verbose=False):
    """
    Returns an OptimizeResult for the maximization problem 
            Q(Xt,Z) s.t. alpha(Xt,Z) == alpha,
    thus controlling for Type I error.
    """
    # Generate lattice points
    lattice_points = van_der_corput(Nmax, scramble=True, seed=random_seed)
    lattice_points_col = lattice_points.reshape(-1, 1)
    t, r, Is = bsampler.t, bsampler.r, bsampler.Is
    Sigma_0 = bsampler.Sigma_0
    Delta_Xt = bsampler.sample_Delta_posterior(Xt=Xt, N=N_DeltaXt)

    # Solve using minimization solver
    start_time = time.time()
    sqp_minimizer = solver(
        power=lambda Z: np.array([
            power(Z=cp.array(Z), Xt=Xt, t=t, r=r, Is=Is,
                      Sigma_0=Sigma_0, Delta_Xt=Delta_Xt,
                      Nmax=NMAX, lattice_points_col=lattice_points_col,
                      slice_size=SLICE_SIZE).get()]),
            constraint=lambda Z: np.array([
                constraint(Z=cp.array(Z), Xt=Xt, t=t, alpha=ALPHA,
                           Sigma_0=Sigma_0, Nmax=NMAX, lattice_points=lattice_points).get()]))
    res = sqp_minimizer.minimize(x0=initial_Z, verbose=verbose)

    return res


##################################
### case of 1d
##################################
def power_1d(Z, Xt, t=0.5, r=0.8, Is=211.0,
             Sigma_0=1, Delta_Xt=None, degenerate=None,
             lattice_points_col=None,
             Nmax=6000, slice_size=10000):
    """
    Returns a computed power based on the power function Q(Z,Xt) for 1D case.

    Parameters
    ----------
    Z : cupy.ndarray(1)
        The threshold value of the entire population
    Xt : cupy.ndarray(1)
        A single sample from posterior distribution Xₜ
    t : float
        'time', the point in the trial the test takes place
        (default is 0.5)
    r : float
        Subgroup proportion vs whole
        (default is 0.9)
    Is : float
        Total number of information units
        (default is 211)
    Sigma_0 : float
        Variance of the normal distribution 
    Delta_Xt : cupy.ndarray(N_DeltaXt, 1)
        Vector of samples from posterior distribution Xₜ
    lattice_points_col : cupy.ndarray(Nmax, 1)
        Vector of lattice points in columnar form
    Nmax : int
        Number of lattice points
    slice_size : int
        Maximum size of vector to handle concurrently
        This parameter can be raised or lowered based on GPU performance and memory capacity
        (default is 10000)
    degenerate : integer
        none if the model is non-degenerate
        0 if the model only considers subgroup
        1 if the model only considers whole population
    """
    sqrt1mt = cp.sqrt(1 - t)
    if degenerate == 0:  
        ZXt = (Z - cp.sqrt( (r*t) / (1-(1-r)*t) ) * Xt) / cp.sqrt( (1-t) / (1-(1-r)*t) ) 
        # sqrt1mt_sub = (cp.sqrt(1 - t + r*t) - t * cp.sqrt(r)) / sqrt1mt
        # Deltat = Delta_Xt * sqrt1mt_sub * cp.sqrt(Is)
        Deltat = Delta_Xt * sqrt1mt *cp.sqrt(Is)
    elif degenerate == 1:  # whole popu
        ZXt = (Z - cp.sqrt(t) * Xt) / sqrt1mt
        Deltat = Delta_Xt * sqrt1mt * cp.sqrt(Is)
    # Compute the CDF for all elements
    cdf_values = cdf1D(ZXt - Deltat, mean=0.0, std=cp.sqrt(Sigma_0))
    result = cp.mean(cdf_values)

    # Return the power
    return result.item()  # 1 - result.item()  # return CDF rather than power based on 2D code


def constraint_1d(Z, Xt, t=0.5, r=0.8, Sigma_0=1,
                  lattice_points=None, degenerate=None,
                  Nmax=6000, alpha=0.025):
    """
    Returns a computed power based on the constraint function alpha(Z,Xt) for 1D case.

    Parameters
    ----------
    Z : cupy.ndarray(1)
        The threshold value of the entire population
    Xt : cupy.ndarray(1)
        A single sample from posterior distribution Xₜ
    t : float
        'time', the point in the trial the test takes place
        (default is 0.5)
    r : float
        Subgroup proportion vs whole
        (default is 0.9)
    Sigma_0 : float
        Variance of the normal distribution
    lattice_points : cupy.ndarray(Nmax, 1)
        Vector of lattice points
    alpha : float, optional
        The target Type I error
        (default is 0.025)
    degenerate : integer
        none if the model is non-degenerate
        0 if the model only considers subgroup
        1 if the model only considers whole population
    """
   
    sqrt1mt = cp.sqrt(1 - t)
    if degenerate == 0:  # subgroup
        ZXt = (Z - cp.sqrt( (r*t) / (1-(1-r)*t) ) * Xt) / cp.sqrt( (1-t) / (1-(1-r)*t) ) 
    elif degenerate == 1:  # whole popu
        ZXt = (Z - cp.sqrt(t) * Xt) / sqrt1mt
    res = cdf1D(ZXt, mean=0.0, std=cp.sqrt(Sigma_0)).item()
    return -1. + alpha + res


    

def get_optimalPower_1d(t=0.25, r=1, Is=211, p_1=0.25,
                        delta=0.2, d=0.0, random_seed=101,
                        Sigma_1=1, Sigma1_coeff=0.2,
                        Xt=cp.array([2.1]), alpha=0.025, degenerate = None,
                        Nmax=6000, N_DeltaXt=6000, slice_size=20000,
                        initial_Z=cp.array([1.]), solver=None, verbose=False):
    """
    Returns an OptimizeResult for the maximization problem Q(Xt,Z) s.t. alpha(Xt,Z) == alpha,
    thus controlling for Type I error.
    """
    # Initialize BayesSampler with 1D parameters
    bsampler = BayesSampler(t=t, r=r, Is=Is, p_1=p_1,
                            delta=delta, d=d, Sigma_1=Sigma_1, Sigma1_coeff=Sigma1_coeff,
                            degenerate=degenerate,
                            random_seed=random_seed)

    # Solve using the specified solver
    start_time = time.time()
    sqp_minimizer = solver(
        power=lambda Z: power_1d(Z=cp.array(Z), Xt=Xt, t=t, r=r, Is=Is,
                                 Sigma_0=bsampler.Sigma_0,
                                 Delta_Xt=bsampler.sample_Delta_posterior(Xt=Xt, N=N_DeltaXt)),
        constraint=lambda Z: constraint_1d(Z=cp.array(Z), Xt=Xt, t=t,
                                           Sigma_0=bsampler.Sigma_0, alpha=alpha)
    )
    res = sqp_minimizer.minimize(x0=initial_Z, verbose=verbose)
    end_time = time.time()

    tabular_results = {
        'Z': f"({res['x'][0]})",
        'power': 1 - res['fun'],
        'alpha': alpha - constraint_1d(cp.array(res['x']), Xt=Xt, t=t, Sigma_0=bsampler.Sigma_0, alpha=alpha),
        'runtime': round(end_time - start_time, 6),
    }

    return tabular_results
