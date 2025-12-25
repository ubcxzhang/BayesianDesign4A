import unittest
import cupy as cp
import numpy as np
from scipy.stats import multivariate_normal
from gbayesdesign.mvn import van_der_corput, pdf2D, cdf2DLattice, redVecCDF2DLattice

class TestMVNmethods(unittest.TestCase):
    def test_van_der_corput(self):
        lattice_points = van_der_corput(n=6, scramble=False)
        cp.testing.assert_array_equal(
            lattice_points,
            cp.array([0., 0.5, 0.25, 0.75, 0.125, 0.625]))
    
    def test_pdf2D(self):
        x_cpu = np.array([0.,0.])
        mean_cpu = np.array([0.,0.])
        cov_cpu = np.array([[1., 0.],[0.,1.]])
        scipy_res = multivariate_normal.pdf(x=x_cpu, mean=mean_cpu, cov=cov_cpu)

        x_gpu = cp.asarray(x_cpu)
        mean_gpu = cp.asarray(mean_cpu)
        cov_gpu = cp.asarray(cov_cpu)
        cupy_res = pdf2D(mu=mean_gpu, Sigma=cov_gpu, x=x_gpu)

        cp.testing.assert_allclose(cupy_res, scipy_res, rtol=1e-03)

    def test_cdf2DLattice(self):
        """
        Compares custom single multivariate normal CDF implementation 
        against scipy.stats.multivariate_normal for reasonable accuracy
        """
        sigma_cpu = np.array([[2.0, 0.3], [0.3, 0.5]])
        ub_cpu = np.array([1.,2.])
        scipy_cdf = multivariate_normal([0.0, 0.0], sigma_cpu)
        scipy_res = scipy_cdf.cdf(ub_cpu)

        lattice_points = van_der_corput(n=6000, scramble=False, seed=101)

        sigma_gpu = cp.asarray(sigma_cpu)
        ub_gpu = cp.asarray(ub_cpu)
        cupy_res = cdf2DLattice(sigma=sigma_gpu, ub=ub_gpu, 
            Nmax=6000, lattice_points=lattice_points)
        
        cp.testing.assert_allclose(cupy_res, scipy_res, rtol=1e-03)

    def test_redVecCDF2DLattice(self):
        """
        Compares custom vectorized multivariate normal CDF implementation 
        against scipy.stats.multivariate_normal for reasonable accuracy
        """
        sigma_cpu = np.array([[2.0, 0.3], [0.3, 0.5]])
        b_values = np.array([[0.,0.],[1.,2.],[2.,1.],[1.,2.]])
        b_cpu = np.repeat(b_values, 10000, axis=0)
        scipy_cdf = multivariate_normal([0.0, 0.0], sigma_cpu)
        scipy_res = scipy_cdf.cdf(b_cpu).mean()

        lattice_points_col = van_der_corput(n=6000, scramble=False, seed=101).reshape(-1,1)

        sigma_gpu = cp.asarray(sigma_cpu)
        b_gpu = cp.asarray(b_cpu)
        cupy_res = redVecCDF2DLattice(sigma=sigma_gpu, ub=b_gpu, 
            Nmax=6000, lattice_points=lattice_points_col,
            slice_size=10000)
        
        cp.testing.assert_allclose(cupy_res, scipy_res, rtol=1e-03)

if __name__ == '__main__':
    unittest.main()