import unittest
import cupy as cp
from gbayesdesign.BayesSampler import BayesSampler

class TestBayesSampler(unittest.TestCase):
    def setUp(self):
        self.bsampler = BayesSampler(
            t=0.5, r=0.5, Is=211, p_1=0.75, 
            delta=0.36, d=0.24,
            Sigma1_coeff=0.01, random_seed=101)
        self.Xt = cp.array([1.,2.])

    def test_Sigma0_value(self):
        cp.testing.assert_array_almost_equal(
            self.bsampler.Sigma_0,
            cp.array([[1., 0.70710678], [0.70710678, 1.]]))

    def test_Sigma1_value(self):
        cp.testing.assert_array_almost_equal(
            self.bsampler.Sigma_1,
            cp.array([[0.01,0.00707107], [0.00707107,0.01]]))
    
    def test_mu1_value(self):
        cp.testing.assert_array_equal(
            self.bsampler.mu_1,
            cp.array([0.36, 0.36]))
        
    def test_mu1p_value(self):
        cp.testing.assert_array_equal(
            self.bsampler.mu_1p,
            cp.array([0.36, 0.6]))
    
    def test_Sigma2_value(self):
        cp.testing.assert_array_almost_equal(
            self.bsampler.Sigma_2,
            cp.array([[0.00472981,0.00382288], [0.00382288,0.00636318]]))

    def test_mu2_value(self):
        mu_2, mu_2p, p_2 = self.bsampler.get_posteriorVar(self.Xt)
        cp.testing.assert_array_almost_equal(
            mu_2,
            cp.array([0.21191084, 0.30178063]),
            decimal=6)
    
    def test_mu2p_value(self):
        mu_2, mu_2p, p_2 = self.bsampler.get_posteriorVar(self.Xt)
        cp.testing.assert_array_almost_equal(
            mu_2p,
            cp.array([0.2348742, 0.47746031]),
            decimal=6)
    
    def test_p2_value(self):
        mu_2, mu_2p, p_2 = self.bsampler.get_posteriorVar(self.Xt)
        cp.testing.assert_array_almost_equal(
            p_2, cp.array([0.80170845]),
            decimal=6)
        
    def test_mu3p_value(self):
        cp.testing.assert_array_almost_equal(
            self.bsampler.mu_3p,
            cp.array([3.69767495, 4.35775171]),
            decimal=6)

    def test_Sigma3_value(self):
        cp.testing.assert_array_almost_equal(
            self.bsampler.Sigma_3,
            cp.array([[2.055,1.23460678], [1.23460678,1.5275]]))
    
    def test_mu3_value(self):
        cp.testing.assert_array_almost_equal(
            self.bsampler.mu_3,
            cp.array([3.69767495, 2.61465103]),
            decimal=6)
        
    def test_mu3p_value(self):
        cp.testing.assert_array_almost_equal(
            self.bsampler.mu_3p,
            cp.array([3.69767495, 4.35775171]),
            decimal=6)
    
    def test_priorDistribution_value(self):
        cp.testing.assert_array_almost_equal(
            self.bsampler.Sigma_2,
            cp.array([[0.00472981,0.00382288], [0.00382288,0.00636318]])) 

if __name__ == '__main__':
    unittest.main()