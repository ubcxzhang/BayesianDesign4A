import cupy as cp
from gbayesdesign.mvn import pdf2D, pdf1D
from . import rndgenerator


class BayesSampler:
    """
    Takes input parameters and calculates new parameters and random samples for:
    1) a prior distribution of drug efficacy 𝚫,
    2) a posterior distribution based on the prior distribution 𝚫|Xₜ, and
    3) a marginal interim distribution at t, Xₜ

    Attributes
    ----------
    t : float
        The point in the trial the test takes place, "time"
        Input parameter
        (default is 0.5)
    r : float
        Subgroup proportion vs whole
        Input parameter
        (default is 0.5)
    Is : float
        Total number of information units
        Input parameter
        (default is 211)
    p_1 : float
        The prior probability of a subgroup effect existing
        Input parameter
        (default is 0.75)
    delta : float
        The mean of drug efficacy in the entire population
        Input parameter
    d : float
        The degree of additional subgroup effect
        Input parameter
    Sigma_0 : cupy.ndarray(2,2)
        The covariance matrix of the distribution
    Sigma_1 : cupy.ndarray(2,2)
        The covariance matrix of the prior distribution
        Can either be defined as an input parameter 
        or generated from Sigma_0 using an input coefficient
    Sigma_2 : cupy.ndarray(2,2)
        The covariance matrix of generated posterior distributions
    Sigma_3 : cupy.ndarray(2,2)
        The covariance matrix of the marginal interim distribution
    mu_1 : cupy.ndarray(2)
        The mean efficacy on the entire population, generated from input delta
    mu_1p : cupy.ndarray(2)
        The mean efficacy on the subgroup, generated from a combination of inputs delta and d
    mu_3 : cupy.ndarray(2)
        The mean efficacy on the entire population in the interim distribution
    mu_3p : cupy.ndarray(2)
        The mean efficacy on the subgroup population in the interim distribution
    degenerate : integer
        none if the model is non-degenerate
        0 if the model only considers subgroup
        1 if the model only considers whole population
    """

    def __init__(self, t=0.5, r=0.5, Is=211.0, p_1=0.75,
                 delta=0.36, d=0.24,
                 Sigma_1=None, Sigma1_coeff=0.2,
                 random_seed=101, degenerate=None):
        """
        Parameters
        ----------
        self
        t : float
            The point in the trial the test takes place
            (default is 0.5)
        r : float
            Subgroup proportion vs whole
            (default is 0.5)
        Is : float
            Total number of information units
            (default is 211)
        p_1 : float
            The prior probability of a subgroup effect existing
            (default is 0.75)
        delta : float
            The mean of drug efficacy in the entire population
        d : float
            The degree of additional subgroup effect
        Sigma_1 : cupy.ndarray(2,2), optional
            The covariance matrix of the prior distribution
            If left as None, the constructor will use Sigma1_coeff * Sigma_0 instead
            (default is None)
        Sigma1_coeff : float, optional
            The coefficient in Sigma_1 = Sigma1_coeff * Sigma_0
            (default is 0.01)
        random_seed : int, optional
            Seed for random number generator
            Set the seed for reproducible results
            (default is 101)
        degenerate : integer
            none if the model is non-degenerate 
            0 if the model only considers subgroup
            1 if the model only considers whole population
        """
        self.t, self.r, self.Is, self.p_1 = t, r, Is, p_1
        self.delta, self.d, self.Sigma1_coeff = delta, d, Sigma1_coeff
        self.degenerate = degenerate
        # Set covariance matrix of all test information units, 𝚺₀
        if self.degenerate is None:
            # Set covariance matrix of all test information units, 𝚺₀
            self.Sigma_0 = cp.identity(2)
            self.Sigma_0[1, 0] = self.Sigma_0[0, 1] = cp.sqrt(r);

            # Set prior covariance matrix 𝚺₁
            if Sigma_1 is not None:
                self.Sigma_1 = Sigma_1
            else:
                self.Sigma_1 = Sigma1_coeff * self.Sigma_0

            # Set prior mean of efficacy for whole population 𝛍₁
            self.mu_1 = cp.array([delta, delta])

            # Set prior mean of efficacy for subgroup 𝛍₁ʹ
            self.mu_1p = cp.array([delta, delta + d])

            # Calculate A using Is, r, and t
            self.__A = cp.diag(cp.array([cp.sqrt(Is * t), cp.sqrt(r * Is * t)]))
            # Store inverse of 𝚺₀ and 𝚺₁ for calculation expediency
            self.__Sigma_1Inv = cp.linalg.pinv(self.Sigma_1)
            self.__Sigma_0Inv = cp.linalg.pinv(self.Sigma_0)

            # Compute posterior covariance matrix 𝚺₂
            self.Sigma_2 = cp.linalg.pinv(self.__Sigma_1Inv + self.__A @ self.__Sigma_0Inv @ self.__A)
            if r == 1:  ###################################################################### non-singularity
                self.Sigma_2 = cp.diag(cp.array([Sigma1_coeff / (1 + Sigma1_coeff * self.Is * self.t), Sigma1_coeff / (
                            1 + Sigma1_coeff * self.r * self.Is * self.t)])) @ self.Sigma_0
                # COMPUTE INTERIM DISTRIBUTION
            S0iAS2 = self.__Sigma_0Inv @ self.__A @ self.Sigma_2
            S0iAS2_S1i = S0iAS2 @ self.__Sigma_1Inv

            # Compute marginal interim covariance matrix 𝚺₃ at t
            self.Sigma_3 = cp.linalg.pinv(self.__Sigma_0Inv - (S0iAS2 @ self.__A @ self.__Sigma_0Inv))

            # Compute marginal mean of efficacy for whole population 𝛍₁
            self.mu_3 = self.Sigma_3 @ S0iAS2_S1i @ self.mu_1
            # Compute marginal mean of efficacy for subgroup 𝛍₁ʹ

            self.mu_3p = self.Sigma_3 @ S0iAS2_S1i @ self.mu_1p
        else:
            ## here Sigmas are all variance 
            ## but in side the rs.normal function we should use standard deviation
            if self.degenerate == 0:  # <<<<<<<<<<<<<<<< 1D subgroup
                self.mu_0 = cp.sqrt(self.r * self.Is * self.t) * self.delta
                self.Sigma_0 = cp.array([1])
                if Sigma_1 is not None:
                    self.Sigma_1 = Sigma_1[-1, -1]
                    self.Sigma1_coeff = self.Sigma_1
                else:
                    self.Sigma_1 = Sigma1_coeff * self.Sigma_0

                self.mu_1, self.mu_1p = self.delta, self.delta + self.d

                self.Sigma_2 = Sigma1_coeff / (1 + Sigma1_coeff * self.r * self.Is * self.t)
                self.mu_3, self.mu_3p = (self.delta) * cp.sqrt(self.r * self.Is * self.t), (
                            self.delta + self.d) * cp.sqrt(self.r * self.Is * self.t)
                self.Sigma_3 = 1 + Sigma1_coeff * self.r * self.Is * self.t
                # print("!!!!!!!!!!!!!!!!!!check!!!!!!!!!!!!!!!!")
                # print(f"𝛍3={self.mu_3},{self.Sigma_3}")
            elif self.degenerate == 1:
                self.mu_0 = cp.sqrt(self.Is * self.t) * self.delta
                self.Sigma_0 = cp.array([1])
                if Sigma_1 is not None:
                    self.Sigma_1 = Sigma_1[0, 0]
                    self.Sigma1_coeff = self.Sigma_1
                else:
                    self.Sigma_1 = Sigma1_coeff * self.Sigma_0
                self.mu_1, self.mu_1p = self.delta, self.delta
                self.Sigma_2 = Sigma1_coeff / (1 + Sigma1_coeff * self.Is * self.t)
                self.mu_3, self.mu_3p = (self.delta) * cp.sqrt(self.Is * self.t), (self.delta) * cp.sqrt(
                    self.Is * self.t)
                self.Sigma_3 = 1 + Sigma1_coeff * self.Is * self.t
        # Seed pseudorandom number generator
        rndgenerator.seed(random_seed)

    def sample_Delta(self, N=10000):
        """Return N samples from the prior distribution 𝚫"""
        pvals = cp.array([1 - self.p_1, self.p_1])
        rs = rndgenerator.get_random_state()
        if self.degenerate is None:
            choices = rs.choice(2, p=pvals, size=N).reshape(-1, 1)
            prior_samples = rs.multivariate_normal(self.mu_1p, ensure_positive_definite(self.Sigma_1),
                                                   size=N) * choices + \
                            rs.multivariate_normal(self.mu_1, ensure_positive_definite(self.Sigma_1), size=N) * (
                                        1 - choices)
        else:
            if self.degenerate == 0:  # <<<<<<<<<<<<<<<< 1D
                choices = rs.choice(2, p=pvals, size=N).reshape(-1, 1)
                N_array = rs.normal((self.delta), 1, size=N).reshape(-1, 1)
                Np_array = rs.normal((self.delta + self.d), 1, size=N).reshape(-1, 1)
                prior_samples = ((1 - choices) * N_array + choices * Np_array).flatten()
            elif self.degenerate == 1:
                prior_samples = rs.normal((self.delta), 1, size=N)
                # >>>>>>>>>>>>>>> 1D
        return prior_samples

    def sample_Delta_posterior(self, Xt, N=10000):
        """
        Returns N samples of posterior distribution 𝚫|Xₜ, given a sample Xₜ

        Parameters
        ----------
        Xt : cupy.ndarray(2)
            Interim sample
        N : int, optional
            Number of samples to return
            (default is 10000)
        """
        rs = rndgenerator.get_random_state()
        if self.degenerate is None:
            ASigma0invXt = self.__A @ self.__Sigma_0Inv @ Xt
            mu_2 = self.Sigma_2 @ ((self.__Sigma_1Inv @ self.mu_1) + ASigma0invXt)
            mu_2p = self.Sigma_2 @ ((self.__Sigma_1Inv @ self.mu_1p) + ASigma0invXt)
            if self.r == 1:  ###################################################################### non-singularity
                mu_2 = cp.array([(self.delta + self.Sigma1_coeff * cp.sqrt(self.Is * self.t) * Xt[0]) / (
                        1 + self.Sigma1_coeff * self.Is * self.t),
                                 (self.delta + self.Sigma1_coeff * cp.sqrt(self.r * self.Is * self.t) * Xt[1]) / (
                                         1 + self.Sigma1_coeff * self.r * self.Is * self.t)])
                mu_2p = cp.array([(self.delta + self.Sigma1_coeff * cp.sqrt(self.Is * self.t) * Xt[0]) / (
                        1 + self.Sigma1_coeff * self.Is * self.t),
                                  (self.delta + self.d + self.Sigma1_coeff * cp.sqrt(self.r * self.Is * self.t) * Xt[
                                      1]) / (
                                          1 + self.Sigma1_coeff * self.r * self.Is * self.t)])
            # Compute p₂
            f1_Xt = pdf2D(self.mu_3p, ensure_positive_definite(self.Sigma_3), Xt)  # Xₜ if subgroup effect = true
            f0_Xt = pdf2D(self.mu_3, ensure_positive_definite(self.Sigma_3), Xt)  # Xₜ if subgroup effect = false
            p_2 = self.p_1 * f1_Xt / (self.p_1 * f1_Xt + (1 - self.p_1) * f0_Xt)

            pvals = cp.array([1 - p_2, p_2])
            rs = rndgenerator.get_random_state()
            choices = rs.choice(2, p=pvals, size=N).reshape(-1, 1)
            self.Delta_choices = choices.get()
            # posterior_samples = rs.multivariate_normal(mu_2p, self.Sigma_2, size=N) * choices + \
            #                     rs.multivariate_normal(mu_2, self.Sigma_2, size=N) * (1 - choices)
            posterior_samples = rs.multivariate_normal(mu_2p, ensure_positive_definite(self.Sigma_2),
                                                       size=N) * choices + \
                                rs.multivariate_normal(mu_2, ensure_positive_definite(self.Sigma_2), size=N) * (
                                            1 - choices)
        elif Xt.shape == (2,):  # Check if Xt has 2 elements (i.e., shape (2,))
            Xit = Xt[1 - int(self.degenerate)]  # s.t., when self.degenerate=0, Xit = Xt[1] for subgroup model
            if self.degenerate == 0:  # <<<<<<<<<<<<<<<< 1D
                # Xₜ if subgroup effect = true
                f1_Xt = pdf1D(self.mu_3p, self.Sigma_3, Xit)
                # Xₜ if subgroup effect = false
                f0_Xt = pdf1D(self.mu_3, self.Sigma_3, Xit)

                # Compute p₂
                p_2 = self.p_1 * f1_Xt / (self.p_1 * f1_Xt + (1 - self.p_1) * f0_Xt)

                pvals = cp.array([1 - p_2, p_2])
                rs = rndgenerator.get_random_state()
                choices = rs.choice(2, p=pvals, size=N).reshape(-1, 1)
                mu_2 = (self.delta + self.Sigma1_coeff * cp.sqrt(self.r * self.Is * self.t) * Xit) / (
                        1 + self.Sigma1_coeff * self.r * self.Is * self.t)
                mu_2p = (self.delta + self.d + self.Sigma1_coeff * cp.sqrt(self.r * self.Is * self.t) * Xit) / (
                        1 + self.Sigma1_coeff * self.r * self.Is * self.t)

                N_array = rs.normal(mu_2, cp.sqrt(self.Sigma_2), size=N).reshape(-1, 1)
                Np_array = rs.normal(mu_2p, cp.sqrt(self.Sigma_2), size=N).reshape(-1, 1)
                posterior_samples = ((1 - choices) * N_array + choices * Np_array).flatten()

            elif self.degenerate == 1:
                rs = rndgenerator.get_random_state()
                mu_2 = (self.delta + self.Sigma1_coeff * cp.sqrt(self.Is * self.t) * Xit) / (
                        1 + self.Sigma1_coeff * self.Is * self.t)
                posterior_samples = rs.normal(mu_2, cp.sqrt(self.Sigma_2), size=N)
                # >>>>>>>>>>>>>>> 1D
        else:  # Check if Xt has only 1 elements (i.e., shape (1,))
            Xit = Xt
            if self.degenerate == 0:  # <<<<<<<<<<<<<<<< 1D
                # Xₜ if subgroup effect = true
                f1_Xt = pdf1D(self.mu_3p, self.Sigma_3, Xit)
                # Xₜ if subgroup effect = false
                f0_Xt = pdf1D(self.mu_3, self.Sigma_3, Xit)

                # Compute p₂
                p_2 = self.p_1 * f1_Xt / (self.p_1 * f1_Xt + (1 - self.p_1) * f0_Xt)

                pvals = cp.array([1 - p_2, p_2])
                rs = rndgenerator.get_random_state()
                choices = rs.choice(2, p=pvals, size=N).reshape(-1, 1)
                mu_2 = (self.delta + self.Sigma1_coeff * cp.sqrt(self.r * self.Is * self.t) * Xit) / (
                        1 + self.Sigma1_coeff * self.r * self.Is * self.t)
                mu_2p = (self.delta + self.d + self.Sigma1_coeff * cp.sqrt(self.r * self.Is * self.t) * Xit) / (
                        1 + self.Sigma1_coeff * self.r * self.Is * self.t)

                N_array = rs.normal(mu_2, cp.sqrt(self.Sigma_2), size=N).reshape(-1, 1)
                Np_array = rs.normal(mu_2p, cp.sqrt(self.Sigma_2), size=N).reshape(-1, 1)
                posterior_samples = ((1 - choices) * N_array + choices * Np_array).flatten()

            elif self.degenerate == 1:
                rs = rndgenerator.get_random_state()
                mu_2 = (self.delta + self.Sigma1_coeff * cp.sqrt(self.Is * self.t) * Xit) / (
                        1 + self.Sigma1_coeff * self.Is * self.t)
                posterior_samples = rs.normal(mu_2, cp.sqrt(self.Sigma_2), size=N)
                # >>>>>>>>>>>>>>> 1D
        return posterior_samples

    def sample_Xt(self, N=10000):
        """
        Return N samples of marginal interim distribution Xₜ|i
        conditional on whether subgroup effect i exists
        (independent of drug efficacy)
        """
        if self.degenerate is None:
            pvals = cp.array([1 - self.p_1, self.p_1])
            rs = rndgenerator.get_random_state()
            choices = rs.choice(2, p=pvals, size=N).reshape(-1, 1)
            XtMarginal_samples = rs.multivariate_normal(self.mu_3, ensure_positive_definite(self.Sigma_3), size=N) * (
                        1 - choices) + \
                                 rs.multivariate_normal(self.mu_3p, ensure_positive_definite(self.Sigma_3),
                                                        size=N) * choices
            self.xt_choices = choices.get()
        else:
            # 1D
            rs = rndgenerator.get_random_state()  # <- add here
            if self.degenerate == 0:  # <<<<<<<<<<<<<<<< 1D
                pvals = cp.array([1 - self.p_1, self.p_1])
                rs = rndgenerator.get_random_state()
                choices = rs.choice(2, p=pvals, size=N).reshape(-1, 1)
                N_array = rs.normal(self.mu_3, cp.sqrt(self.Sigma_3), size=N).reshape(-1, 1)
                Np_array = rs.normal(self.mu_3p, cp.sqrt(self.Sigma_3), size=N).reshape(-1, 1)
                XtMarginal_samples = ((1 - choices) * N_array + choices * Np_array).flatten()
            elif self.degenerate == 1:
                XtMarginal_samples = rs.normal(self.mu_3, cp.sqrt(self.Sigma_3), size=N)
            # >>>>>>>>>>>>>>> 1D

        return XtMarginal_samples

    def get_posteriorVar(self, Xt):
        """
        Returns the values of mu_2, mu_2 and p_2 given Xt,
        for testing purposes

        Parameters
        ----------
        Xt : cupy.ndarray(2)
            Interim sample Xₜ at test time t
        or
        Xt_r : cupy.ndarray(1)
            Interim sample Xₜ at test time t only dim = r

        Returns
        -------
        mu_2 : cupy.ndarray(2)
            𝛍₂, Mean of whole population in posterior distribution 𝚫|Xₜ
        mu_2p : cupy.ndarray(2)
            𝛍₂ʹ, Mean of subgroup population in posterior distribution 𝚫|Xₜ
        p_2 : float
            The posterior probability of a subgroup effect existing         
        """
        if self.degenerate is None:
            ASigma0invXt = self.__A @ self.__Sigma_0Inv @ Xt
            mu_2 = cp.dot(
                self.Sigma_2,
                self.__Sigma_1Inv @ self.mu_1 + ASigma0invXt)
            mu_2p = cp.dot(
                self.Sigma_2,
                self.__Sigma_1Inv @ self.mu_1p + ASigma0invXt)
            if self.r == 1:  ###################################################################### non-singularity
                mu_2 = cp.array([(self.delta + self.Sigma1_coeff * cp.sqrt(self.Is * self.t) * Xt[0]) / (
                        1 + self.Sigma1_coeff * self.Is * self.t),
                                 (self.delta + self.Sigma1_coeff * cp.sqrt(self.r * self.Is * self.t) * Xt[1]) / (
                                         1 + self.Sigma1_coeff * self.r * self.Is * self.t)])
                mu_2p = cp.array([(self.delta + self.Sigma1_coeff * cp.sqrt(self.Is * self.t) * Xt[0]) / (
                        1 + self.Sigma1_coeff * self.Is * self.t),
                                  (self.delta + self.d + self.Sigma1_coeff * cp.sqrt(self.r * self.Is * self.t) * Xt[
                                      1]) / (
                                          1 + self.Sigma1_coeff * self.r * self.Is * self.t)])
            # Xₜ if subgroup effect = true
            f1_Xt = pdf2D(self.mu_3p, ensure_positive_definite(self.Sigma_3), Xt)
            # Xₜ if subgroup effect = false
            f0_Xt = pdf2D(self.mu_3, ensure_positive_definite(self.Sigma_3), Xt)

            # Compute p₂
            p_2 = self.p_1 * f1_Xt / (self.p_1 * f1_Xt + (1 - self.p_1) * f0_Xt)
        elif Xt.shape == (2,):  # Check if Xt has 2 elements (i.e., shape (2,))
            Xit = Xt[1 - int(self.degenerate)]
            if self.degenerate == 0:  # <<<<<<<<<<<<<<<< 1D
                mu_2 = (self.delta + self.Sigma1_coeff * cp.sqrt(self.r * self.Is * self.t) * Xit) / (
                        1 + self.Sigma1_coeff * self.r * self.Is * self.t)
                mu_2p = (self.delta + self.d + self.Sigma1_coeff * cp.sqrt(self.r * self.Is * self.t) * Xit) / (
                        1 + self.Sigma1_coeff * self.r * self.Is * self.t)
                f1_Xt = pdf1D(self.mu_3p, self.Sigma_3, Xit)  # only care about X1t
                # Xₜ if subgroup effect = false
                f0_Xt = pdf1D(self.mu_3, self.Sigma_3, Xit)

                # Compute p₂
                p_2 = self.p_1 * f1_Xt / (self.p_1 * f1_Xt + (1 - self.p_1) * f0_Xt)

            elif self.degenerate == 1:
                mu_2 = (self.delta + self.Sigma1_coeff * cp.sqrt(self.Is * self.t) * Xit) / (
                        1 + self.Sigma1_coeff * self.Is * self.t)  # only care about X2t
                mu_2p = mu_2
                f1_Xt = pdf1D(self.mu_3p, self.Sigma_3, Xit)  # only care about X1t
                # Xₜ if subgroup effect = false
                f0_Xt = pdf1D(self.mu_3, self.Sigma_3, Xit)

                # Compute p₂
                p_2 = self.p_1 * f1_Xt / (self.p_1 * f1_Xt + (1 - self.p_1) * f0_Xt)
                # >>>>>>>>>>>>>>> 1D 
            else:
                print("Check degeneration")
        else:
            if self.degenerate == 0:  # <<<<<<<<<<<<<<<< 1D
                mu_2 = (self.delta + self.Sigma1_coeff * cp.sqrt(self.r * self.Is * self.t) * Xt) / (
                        1 + self.Sigma1_coeff * self.r * self.Is * self.t)
                mu_2p = (self.delta + self.d + self.Sigma1_coeff * cp.sqrt(self.r * self.Is * self.t) * Xt) / (
                        1 + self.Sigma1_coeff * self.r * self.Is * self.t)
                # Xₜ if subgroup effect = true
                f1_Xt = pdf1D(self.mu_3p, self.Sigma_3, Xt)  # only care about X1t
                # Xₜ if subgroup effect = false
                f0_Xt = pdf1D(self.mu_3, self.Sigma_3, Xt)

                # Compute p₂
                p_2 = self.p_1 * f1_Xt / (self.p_1 * f1_Xt + (1 - self.p_1) * f0_Xt)

            elif self.degenerate == 1:
                mu_2 = (self.delta + self.Sigma1_coeff * cp.sqrt(self.Is * self.t) * Xt) / (
                        1 + self.Sigma1_coeff * self.Is * self.t)
                mu_2p = mu_2

                f1_Xt = pdf1D(self.mu_3p, self.Sigma_3, Xt)  # only care about X1t
                # Xₜ if subgroup effect = false
                f0_Xt = pdf1D(self.mu_3, self.Sigma_3, Xt)

                # Compute p₂
                p_2 = self.p_1 * f1_Xt / (self.p_1 * f1_Xt + (1 - self.p_1) * f0_Xt)
                # >>>>>>>>>>>>>>> 1D 
            else:
                print("Check degeneration")
        return mu_2, mu_2p, p_2

    def ensure_positive_definite(matrix, jitter=1e-6):
        xp = cp.get_array_module(matrix)
        return matrix + xp.eye(matrix.shape[0]) * jitter