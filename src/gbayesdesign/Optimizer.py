import numpy as np
from scipy import optimize
# from pdfo import pdfo, Bounds, NonlinearConstraint

class Optimizer(object):
    """
    Generic Optimizer object,
    Used to standardize derivative-free constrained nonlinear multivariate solvers 
    from different packages

    Attributes
    ----------
    power : function
        The objective function to be solved for
        Within this context, it is named "power"
    constraint : function
        The constraint function
        Within this context, maintains Type I error at a predetermined threshold
    """
    def __init__(self, power=None, constraint=None):
        self.power = power
        self.constraint = constraint
    def minimize(self,**kwargs):
        """
        Call this function to run the solver
        Implementation is left entirely to child classes to define
        """
        pass


class ZSQP(Optimizer):
    """
    Adaptation of SciPy's SLSQP (Sequential Least SQuares Processing) minimization solver
    to the standard Optimizer interface for the Q(Xt, Z) problem
    """

    def __init__(self, power=None, constraint=None):
        super().__init__(power, constraint)

    def minimize(self, x0=np.array([1., 2.]), lb=0., ub=4.5, verbose=False):
        """
        Call this function to run SciPy's SLSQP solver

        Parameters
        ----------
        x0 : numpy.ndarray(2)
            Initial value of input variable/s
        lb : float
            Lower bound for x
            (Default is 0)
        ub : float
            Upper bound for x
            (Default is 4.5)
        verbose : boolean
            Verbose flag
            Set to True for debugging output
            (Default is False)
        """
        # inequality means that it is to be non-negative: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
                # Define inequality constraint to be non-negative
        nonlin_con_ineq = {'type': 'ineq', 'fun': self.constraint}
        
        # Attempt optimization with increasing upper bounds until success or ub=6
        while True:
            bounds = [(lb, ub), (lb, ub)]
            res = optimize.minimize(
                self.power,
                x0,
                method='SLSQP',
                jac='3-point',
                constraints=[nonlin_con_ineq],
                tol=None,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12,
                    'eps': 1.4901161193847656e-08,
                    'iprint': 100,
                    'disp': verbose
                },
                bounds=bounds
            )
            
            # Check if optimization was successful
            if res.success:
                break  # Exit the loop if successful
            
            # If unsuccessful and ub is less than 6, increment ub
            if ub < 7:
                ub += .25
                if verbose:
                    print(f"Optimization failed. Increasing ub to {ub}.")
            else:
                if verbose:
                    print("Optimization failed and ub reached maximum limit of 6.")
                break  # Stop if ub has reached 6 and optimization still fails
        
        return res

    
class ZSQP_1d(Optimizer):
    """
    Adaptation of SciPy's SLSQP (Sequential Least SQuares Processing) minimization solver
    to the standard Optimizer interface for the Q(Xt, Z) problem
    """

    def __init__(self, power=None, constraint=None):
        super().__init__(power, constraint)

    def minimize(self, x0=1., lb=0., ub=4., verbose=False):
        """
        Call this function to run SciPy's SLSQP solver

        Parameters
        ----------
        x0 : float
            Initial value of input variable
        lb : float
            Lower bound for x
            (Default is 0)
        ub : float
            Upper bound for x
            (Default is 3.5)
        verbose : boolean
            Verbose flag
            Set to True for debugging output
            (Default is False)
        """
        
        nonlin_con_ineq = {'type': 'ineq', 'fun': self.constraint}
        bounds = [(lb, ub)]

        # Now, run the solver
        print("Running solver...")
        res = optimize.minimize(self.power, x0, method='SLSQP',
                                jac='3-point', constraints=(nonlin_con_ineq),
                                tol=None, callback=None, bounds=bounds,
                                options={
                                    'maxiter': 1000, 'ftol': 1e-32,
                                    'eps': 1.4901161193847656e-32,
                                    'finite_diff_rel_step': None,
                                    'iprint': 100, 'disp': verbose})
        print("Solver finished.")
        print("Result:", res)

        # Additional debug info      
        return res  # Return raw OptimizeResult
