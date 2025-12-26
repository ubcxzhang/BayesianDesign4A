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
    # print("%%%%%%%%%%%%%% check %%%%%%%%%%%%%%")
    # print(f"Xt={Xt}, ZXt={ZXt}, Deltat_Xt={Delta_Xt},meanDelta_Xt={Delta_Xt.mean()}")
    # print(f"sigma={Sigma_0}, mean={ZXt - Deltat}, type2err={res}")
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
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
    # res = redVecCDF2DLattice(sigma=Sigma_0, ub=ZXt - Deltat,
    #                          Nmax=Nmax, lattice_points=lattice_points_col, slice_size=slice_size)
    res = cdf2DLattice(sigma=Sigma_0, ub=ZXt - Deltat,
                       Nmax=Nmax, lattice_points=lattice_points)
    # print("%%%%%%%%%%%%%% check %%%%%%%%%%%%%%")
    # print(f"Xt={Xt}, ZXt={ZXt}, Deltat_Xt={Delta_Xt},meanDelta_Xt={Delta_Xt.mean()}")
    # print(f"sigma={Sigma_0}, mean={ZXt - Deltat}, type2err={res}")
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
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
###case of 1d
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
    # Reduce the result to a single scalar
    # Here we assume you want the mean or some aggregation of the result
    # If you need the minimum or some specific statistic, adjust this line
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
   
    # res = cdf2DLattice(sigma=Sigma_0, ub=ZXt,
    #     Nmax=Nmax, lattice_points=lattice_points)
    # print("Shape of constraint_1d output:", (-1.0 + alpha + res).shape)  
    # return -1. + alpha + res #The constraint is now constraint()>=0
    sqrt1mt = cp.sqrt(1 - t)
    if degenerate == 0:  # subgroup
        ZXt = (Z - cp.sqrt( (r*t) / (1-(1-r)*t) ) * Xt) / cp.sqrt( (1-t) / (1-(1-r)*t) ) 
    elif degenerate == 1:  # whole popu
        ZXt = (Z - cp.sqrt(t) * Xt) / sqrt1mt
    res = cdf1D(ZXt, mean=0.0, std=cp.sqrt(Sigma_0)).item()
    # print(f"Constraint value at x={Xt}: {-1. + alpha + res}")
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

def power_integrated_Xt_bayes(
    Z,
    bsampler,                    # BayesSampler
    N_Xt=256,                    # Monte Carlo draws of X_t ~ f(X_t)
    N_DeltaXt=6000,             # posterior draws of Δ|X_t (your usual)
    lattice_points_col=None,
    Nmax=6000,
    slice_size=20000,
    seed=101,
):
    """
    Compute design-stage Bayesian predictive power:
      Q(Z) = E_{X_t}[ E_{Δ|X_t}[ 1 - Φ_Σ0( (Z - √t X_t)/√(1-t) - √(1-t)S Δ ) ] ].
    Reuses your GPU lattice code:
      - inner expectation over Δ|X_t via redVecCDF2DLattice
      - outer expectation over X_t via Monte Carlo using bsampler.sample_Xt
    Returns a Python float in [0,1].
    """
    t, r, Is = bsampler.t, bsampler.r, bsampler.Is
    Sigma_0 = bsampler.Sigma_0

    # lattice
    if lattice_points_col is None:
        lp = van_der_corput(Nmax, scramble=True, seed=seed)
        lattice_points_col = lp.reshape(-1, 1)

    # draw X_t from marginal f(X_t) encoded by BayesSampler (mixture)
    Xt_draws = bsampler.sample_Xt(N=N_Xt)

    vals = []
    for i in range(N_Xt):
        Xt = Xt_draws[i]
        # posterior Δ|X_t samples (shape [N_DeltaXt,2])
        Delta_Xt = bsampler.sample_Delta_posterior(Xt=Xt, N=N_DeltaXt)

        # conditional-on-X_t power using your existing kernel
        val = power(
            Z=cp.asarray(Z), Xt=Xt,
            t=t, r=r, Is=Is, Sigma_0=Sigma_0,
            Delta_Xt=Delta_Xt,
            lattice_points_col=lattice_points_col,
            Nmax=Nmax, slice_size=slice_size
        )
        vals.append(float(val))
    return float(np.mean(vals))

def power_integrated_Xt_bayes_fast(
    Z,
    bsampler,                    # BayesSampler
    N_Xt=256,                    # Monte Carlo draws of X_t ~ f(X_t)
    lattice_points=None,         # 1D van-der-Corput for cdf2DLattice
    Nmax=2048,
    seed=101,
):
    """
    Fast Bayesian predictive type-II error integrating over X_t ONLY.
    Uses closed-form Δ|X_t (mixture with posterior mean/cov) so there is
    NO sampling of Δ|X_t. For each X_t:
       β_i = (1-p2_i) Φ_{Σ_eff}( u_i - √(1-t) S μ2_i )
            +    p2_i  Φ_{Σ_eff}( u_i - √(1-t) S μ2p_i )
    where Σ_eff = Σ0 + (1-t) S Σ2 S^T (constant in Xt).
    Returns a Python float in [0,1] (type-II error). Power = 1 - β.
    """
    t, r, Is = bsampler.t, bsampler.r, bsampler.Is
    Sigma_0 = bsampler.Sigma_0
    S = cp.diag(cp.array([cp.sqrt(Is), cp.sqrt(r * Is)]))
    sqrt1mt = cp.sqrt(1.0 - t)

    # Effective covariance (constant over Xt)
    Sigma_eff = Sigma_0 + (1.0 - t) * (S @ bsampler.Sigma_2 @ S)

    # Lattice for cdf2DLattice (1D array)
    if lattice_points is None:
        lattice_points = van_der_corput(Nmax, scramble=True, seed=seed)

    # Draw X_t from the marginal encoded by BayesSampler
    Xt_draws = bsampler.sample_Xt(N=N_Xt)  # shape (N_Xt, 2)

    Z = cp.asarray(Z)
    vals = []
    for i in range(N_Xt):
        Xt = Xt_draws[i]
        # posterior parameters and mixture weight for THIS Xt
        mu2, mu2p, p2 = bsampler.get_posteriorVar(Xt)

        # u(Xt)
        u = (Z - cp.sqrt(t) * Xt) / sqrt1mt

        # bounds for two mixture components
        ub0 = u - sqrt1mt * (S @ mu2)
        ub1 = u - sqrt1mt * (S @ mu2p)

        # two bivariate CDFs with Σ_eff
        beta0 = cdf2DLattice(sigma=Sigma_eff, ub=ub0, Nmax=Nmax, lattice_points=lattice_points)
        beta1 = cdf2DLattice(sigma=Sigma_eff, ub=ub1, Nmax=Nmax, lattice_points=lattice_points)

        vals.append((1.0 - float(p2)) * float(beta0) + float(p2) * float(beta1))

    return float(np.mean(vals))


def beta_bayes_naive_fast(
    Z,
    bsampler,                   # BayesSampler (for Σ0, I, r, and prior over Δ)
    N_Delta=8000,
    lattice_points=None,        # 1D van-der-Corput (will reshape to column for redVec)
    Nmax=2048,
    slice_size=4096,
    seed=101,
):
    """
    Fast naive (t=0) type-II error:
        β_naive(Z) = E_{Δ ~ prior}[ Φ_{Σ0}( Z - √I S Δ ) ],
    with S = diag(√I, √(rI)).
    Uses redVecCDF2DLattice if available; falls back to per-draw cdf2DLattice.
    Returns Python float (type-II).
    """
    Z = cp.asarray(Z)
    Sigma_0 = bsampler.Sigma_0
    I, r = bsampler.Is, bsampler.r
    scale = cp.asarray([cp.sqrt(I), cp.sqrt(r * I)])

    if lattice_points is None:
        lattice_points = van_der_corput(Nmax, scramble=True, seed=seed)

    # Prior draws of Δ (GPU)
    Delta = bsampler.sample_Delta(N_Delta)  # (N_Delta, 2)
    ub = Z - Delta * scale                  # (N_Delta, 2)

    # Try redVecCDF2DLattice (batched mean); otherwise loop
    try:
        lattice_points_col = lattice_points.reshape(-1, 1)
        beta = redVecCDF2DLattice(
            sigma=Sigma_0,
            ub=ub,
            Nmax=Nmax,
            slice_size=min(slice_size, int(2e4)),
            lattice_points=lattice_points_col
        )
        return float(beta)
    except Exception:
        # Safe fallback: average per-draw cdf2DLattice calls
        vals = []
        for i in range(N_Delta):
            vals.append(float(cdf2DLattice(sigma=Sigma_0, ub=ub[i], Nmax=Nmax, lattice_points=lattice_points)))
        return float(np.mean(vals))


def power_integrated_Xt_bayes_1d(
    Z,
    bsampler,                    # BayesSampler configured with degenerate=0 or 1
    N_Xt=256,
    N_DeltaXt=60000,
    seed=101,
):
    """
    Design-stage predictive power for the 1D (degenerate) branch,
    using your power_1d and bsampler.sample_Xt / sample_Delta_posterior.
    """
    assert bsampler.degenerate in (0, 1), "BayesSampler must be 1D (degenerate=0 or 1)."
    t, r, Is = bsampler.t, bsampler.r, bsampler.Is
    Sigma0_1d = float(bsampler.Sigma_0)  # scalar

    Xt_draws = bsampler.sample_Xt(N=N_Xt)
    vals = []
    for i in range(N_Xt):
        # scalar Xt for the active component
        Xt_scalar = Xt_draws[i] if Xt_draws.ndim == 1 else Xt_draws[i, 1 - int(bsampler.degenerate)]

        # Build a length-2 vector only to satisfy sample_Delta_posterior’s 2D branch:
        if Xt_draws.ndim == 1:
            if bsampler.degenerate == 0:
                Xt_for_post = cp.asarray([0., Xt_scalar])   # subgroup is 2nd component
            else:
                Xt_for_post = cp.asarray([Xt_scalar, 0.])   # whole-pop is 1st component
        else:
            Xt_for_post = Xt_draws[i]

        Delta_Xt = bsampler.sample_Delta_posterior(Xt=Xt_for_post, N=N_DeltaXt)

        # ensure 1D vector for power_1d
        if isinstance(Delta_Xt, cp.ndarray) and Delta_Xt.ndim > 1:
            Delta_Xt = Delta_Xt[:, 1 - int(bsampler.degenerate)]

        val = power_1d(
            Z=cp.asarray(Z), Xt=cp.asarray([Xt_scalar]),
            t=t, r=r, Is=Is, Sigma_0=Sigma0_1d,
            Delta_Xt=cp.asarray(Delta_Xt), degenerate=bsampler.degenerate
        )
        vals.append(float(val))
    return float(np.mean(vals))

def make_Xt_sampler_fixedDelta(Delta_true, t=0.5, r=0.8, Is=211.0, Sigma_0=None, seed=101):
    """
    Returns a callable that draws cp.ndarray (N,2) from X_t ~ N(A_t Δ_true, Σ0).
    """
    rng = np.random.RandomState(seed)
    sqrtIt = np.sqrt(Is * t)
    mean = np.array([sqrtIt, np.sqrt(r) * sqrtIt]) * cp.asnumpy(Delta_true)

    def _sampler(N):
        xs = rng.multivariate_normal(mean=mean, cov=cp.asnumpy(Sigma_0), size=N)
        return cp.asarray(xs)
    return _sampler


def power_integrated_Xt_nb(
    Z,
    Delta_true,                  # cp.ndarray shape (2,)
    t=0.5, r=0.8, Is=211.0,
    Sigma_0=None,
    N_Xt=512,
    lattice_points=None,
    Nmax=6000,
    seed=101,
):
    """
    Design-stage NB power with fixed effect Δ_true:
      Q(Z) = E_{X_t|Δ_true}[ 1 - Φ_Σ0( (Z - √t X_t)/√(1-t) - √(1-t)S Δ_true ) ].
    Uses your cdf2DLattice for each X_t and averages.
    """
    if lattice_points is None:
        lattice_points = van_der_corput(Nmax, scramble=True, seed=seed)

    Xt_sampler = make_Xt_sampler_fixedDelta(Delta_true, t=t, r=r, Is=Is, Sigma_0=Sigma_0, seed=seed)

    vals = []
    sqrt1mt = cp.sqrt(1 - t)
    scale = cp.array([sqrt1mt * cp.sqrt(Is), sqrt1mt * cp.sqrt(r * Is)])
    shift = Delta_true * scale  # √(1-t) S Δ_true

    for Xt in Xt_sampler(N_Xt):
        ZXt = (cp.asarray(Z) - cp.sqrt(t) * Xt) / sqrt1mt
        # type II error at ub = ZXt - shift
        beta = cdf2DLattice(sigma=Sigma_0, ub=ZXt - shift, Nmax=Nmax, lattice_points=lattice_points)
        vals.append(float(beta))
    return float(np.mean(vals))


def power_integrated_Xt_nb_fast(
    Z,
    Delta_true,
    t=0.5, r=0.8, Is=211.0,
    Sigma_0=None,
    N_Xt=512,
    lattice_points=None,
    Nmax=6000,
    seed=101,
):
    if lattice_points is None:
        lattice_points = van_der_corput(Nmax, scramble=True, seed=seed)

    # sample X_t | Δ_true
    rng = np.random.RandomState(seed)
    sqrtIt = np.sqrt(Is * t)
    mean = np.array([sqrtIt, np.sqrt(r) * sqrtIt]) * cp.asnumpy(Delta_true)
    Xt = rng.multivariate_normal(mean=mean, cov=cp.asnumpy(Sigma_0), size=N_Xt)
    Xt = cp.asarray(Xt)

    sqrt1mt = cp.sqrt(1 - t)
    shift = Delta_true * cp.array([sqrt1mt * cp.sqrt(Is), sqrt1mt * cp.sqrt(r*Is)])

    ZXt = (cp.asarray(Z) - cp.sqrt(t) * Xt) / sqrt1mt
    ub = ZXt - shift  # (N_Xt, 2)

    # >>> fix: pass column-shaped lattice points <<<
    lattice_points_col = lattice_points.reshape(-1, 1)

    return float(redVecCDF2DLattice(
        sigma=Sigma_0,
        ub=ub,
        Nmax=Nmax,
        slice_size=min(N_Xt, 20000),
        lattice_points=lattice_points_col
    ))


# def get_PowerZ_bsampler_1d(bsampler=BayesSampler(), random_seed=101,
#                            Xt=cp.array([2.1]), alpha=0.025,
#                            Nmax=6000, N_DeltaXt=60000, slice_size=20000,
#                            initial_Z=np.array([1.]), solver=ZSQP_1d, verbose=False):
#     """
#     Returns an OptimizeResult for the maximization problem 
#             Q(Xt,Z) s.t. alpha(Xt,Z) == alpha,
#     thus controlling for Type I error.
#     """
#     t, r, Is = bsampler.t, bsampler.r, bsampler.Is
#     Sigma_0 = bsampler.Sigma_0
#     Delta_Xt = bsampler.sample_Delta_posterior(Xt=Xt, N=N_DeltaXt)

#     # Solve using minimization solver
#     start_time = time.time()
#     sqp_minimizer = solver(
#         power=lambda Z: np.array([
#             power_1d(Z=cp.array(Z), Xt=Xt, t=t, r=r, Is=Is,
#                      Sigma_0=Sigma_0, Delta_Xt=Delta_Xt)]),
#         constraint=lambda Z: np.array([
#             constraint_1d(Z=cp.array(Z), Xt=Xt, t=t,
#                           Sigma_0=Sigma_0, alpha=alpha)]))
#     res = sqp_minimizer.minimize(x0=initial_Z, verbose=verbose)

    # return res

# if __name__ == "__main__":
#     # Important Z values for different confidence levels
#     ALPHA = 0.025
#     X = cp.array([1.43953147, 3.89059189]) ; X1 = 3.89059189
#     t = 0.25; r=0.5; Is=211; p_1=0.25; Sigma1_coeff=0.2; delta=0.2; d=0
#     NMAX = 6000;
#     SLICE_SIZE = 20000;
#     N_DELTAXT = 80000;
#     SEED = 101
#     lattice_points = van_der_corput(NMAX, scramble=True, seed=SEED)
#     lattice_points_col = lattice_points.reshape(-1, 1)
#         #print(lattice_points)
#         ### Evaluation at many poin
#     i_degenerate = 0
#     Z1 = np.sqrt(1-t) * norm.ppf(1-ALPHA) + np.sqrt(t) * np.array(X1);
#     Z = cp.array([1, 1.5]); Z1_2 = 1.5
#     print(f"At x = {X1}:Z1 for degenerate model are:{Z1}(true), {Z1_2}(opt)")
#     bsampler = BayesSampler(t=t, r=r, Is=Is, p_1=p_1, delta=delta, d=d,
#                                     Sigma1_coeff=Sigma1_coeff, random_seed=SEED)
#     bsampler2 = BayesSampler(t=t, r=r, Is=Is, p_1=p_1, delta=delta, d=d,
#                                 Sigma1_coeff=Sigma1_coeff, degenerate=i_degenerate, random_seed=SEED)
#     Sigma_0 = bsampler.Sigma_0
#     Sigma_02 = bsampler2.Sigma_0
#     Delta_Xt2 = bsampler2.sample_Delta_posterior(X, N_DELTAXT).reshape(-1, 1)
#     print(f"Sigma_0: {Sigma_02}")
#     print(f"Constraint for original model: {ALPHA - constraint(Z=Z, Xt=X, t=t, alpha=ALPHA, Sigma_0=Sigma_0, Nmax=NMAX, lattice_points=lattice_points)}")
#     print(f"Constraint for true deg. model: {ALPHA - constraint_1d(Z=Z1, Xt=X1, t=t, alpha=ALPHA, Sigma_0=1, Nmax=NMAX, lattice_points=lattice_points)}")
#     print(f"Constraint for cal. deg. model: {ALPHA - constraint_1d(Z=Z1_2, Xt=X1, t=t, alpha=ALPHA, Sigma_0=1, Nmax=NMAX, lattice_points=lattice_points)}")

#     sqp_minimizer = get_PowerZ_bsampler(bsampler=bsampler, random_seed=101,
#                         Xt=cp.array(X))
    
#     pdfo_minimizer2 = PDFO_1d(
#                 power=lambda Z: np.array([
#                     power_1d(Z=cp.array(Z1_2), Xt=X1, t=t, r=r, Is=Is,
#                              Sigma_0=Sigma_02, Delta_Xt=Delta_Xt2, degenerate=i_degenerate,
#                              Nmax=NMAX, lattice_points_col=lattice_points_col,
#                              slice_size=SLICE_SIZE)]),
#                 constraint=lambda Z: np.array([
#                     constraint_1d(Z=cp.array(Z1_2), Xt=X1, t=t, alpha=ALPHA,
#                                   Sigma_0=Sigma_02, Nmax=NMAX, lattice_points=lattice_points)]))
#     sqp_res = sqp_minimizer.minimize(x0=cp.asnumpy(Z))
#     pdfo_res2 = pdfo_minimizer2.minimize(x0=cp.asnumpy(sqp_res['x'][1]))
#     print(f'SLSQP results are:\n {str(sqp_res)}')
#     print(f'PDFO  results are:\n {str(pdfo_res2)}') 
