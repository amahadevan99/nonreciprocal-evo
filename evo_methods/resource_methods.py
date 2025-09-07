import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

# samples vector v from i.i.d standard normals, conditional on v . nu_in > ups. See
# "Sampling the Multivariate Standard Normal Distribution under a Weighted Sum Constraint" by Frederic Vrins (2018).
def sample_truncated(nu_in,ups):
    nu = nu_in.copy()
    reg_idx = np.abs(nu)<1e-7
    # regularization step: needed to prevent last random variable from being too large if nu_in[-1] is small
    nu[reg_idx] = 1e-7*np.sign(nu)[reg_idx]
    norm_nu = np.sqrt(np.sum(nu**2))

    if ups/norm_nu<3:
        u = np.random.random()
        cbar = norm_nu*norm.ppf((1-u)+u*norm.cdf(ups/norm_nu))
    else:
        cbar = ups + norm_nu**2/ups*np.random.exponential() # approximation for gaussian sampled deep into tail
    return sample_c(nu,cbar,norm_nu=norm_nu)

# samples x_vec from i.i.d standard normals, conditional on x_vec . nu = c.
def sample_c(nu,c,norm_nu=None):
    n = len(nu)
    if norm_nu is None:
        norm_nu = np.sqrt(np.sum(nu**2))

    if n==1:
        return np.array([c/norm_nu])

    mu = c*nu**2/norm_nu**2
    cov_mat = -np.outer(nu**2,nu**2)
    for i in range(n):
        cov_mat[i,i] = nu[i]**2*(norm_nu**2-nu[i]**2)
    cov_mat = cov_mat/norm_nu**2

    x_vec = np.random.multivariate_normal(mu[:-1],cov_mat[:-1,:-1])
    x_vec = np.append(x_vec,c-np.sum(x_vec))
    return x_vec/nu


class ResourceEvo:
    def __init__(self,filename,K0,D,kappa,mu,sigma,omega,m,Kvec,mut_corr,num_epochs,sample_rate=1,seed=None,check_stability=True,
        max_iter=10,epoch_save_num=10000):
        if seed is None:
            self.seed = np.random.randint(4294967296)
        else:
            self.seed = seed
        np.random.seed(seed=self.seed)
        self.filename = filename # file to save data as npz
        self.K0 = K0 # number of initial strains
        self.D = D # number of resources
        self.kappa = kappa # correlation between random parts of feeding and growth rates
        self.mu = mu # mean of feeding and growth rates
        self.sigma = sigma # stdev of random parts of feeding and growth rates
        self.omega = omega # resource decay rate
        self.m = m # consumer death rate
        self.Kvec = Kvec # resource supply vector
        self.mut_corr = mut_corr # correlation between parent and mutant phenotypes
        self.num_epochs = num_epochs # number of epochs of evolution
        self.idx_list = np.arange(K0) # unique strain indices
        self.sample_rate = sample_rate # subsampling parameter for long simulations
        self.count = K0 # index of next strain to be added
        self.epoch_save_num = epoch_save_num # how often to save data to file
        self.max_iter = max_iter # maximum number of iterations to add back invadable strains when finding fixed point
        self.check_stability = check_stability # whether or not to check stability of fixed points

        a = sigma*np.random.randn(K0,D) # deviations in growth matrix
        b = kappa*a + sigma*np.sqrt(1-kappa**2)*np.random.randn(K0,D) # deviations in feeding matrix

        self.G = mu + a # growth matrix
        self.F = mu + b # feeding matrix

        self.L_list = [] # diversity over time
        self.max_iter_epochs = [] # epochs in which uninvadable fixed point not found
        self.unstable_epochs = [] # epochs in which fixed point is unstable
        self.GK_list = [] # general fitnesses over time
        self.Fn_list = [] # F^T . n over time
        self.coex_list = [] # whether or not parent and mutant coexist
        self.n_list = [] # strain abundances over time
        self.R_list = [] # resource abundances over time
        self.strain_idx_list = [] # strain identifiers over time
        self.epoch_list = [] # epochs in which data were saved
        self.inv_prob = [] # log invasion probability over time

    # finds the fixed point for the strain abundances, under the dynamics of externally supplied resources,
    # using fsolve to solve for the abundances at the fixed point.
    def find_fp_ext(self,ii,n0=None):
        G,F,Kvec,m,omega = self.G,self.F,self.Kvec,self.m,self.omega
        max_iter = self.max_iter
        L = G.shape[0]
        # growth rates of the strains when the abundances are n at idxs, and 0 everywhere else
        # must have len(idxs) = len(n)
        def growth_rates(n,idxs):
            tmp = np.dot(G,Kvec/(omega+np.dot(F[idxs,:].T,n)))-m
            return tmp

        # returns only the growth rates for the strains at idxs
        def growth_rates_solve(n,idxs):
            tmp = np.dot(G,Kvec/(omega+np.dot(F[idxs,:].T,n)))-m
            return tmp[idxs]

        extinct_idxs = np.array([],dtype=int)
        extant_idxs = np.arange(L,dtype=int)
        
        # initial guess for fsolve
        if n0 is None:
            n_init = np.ones(len(extant_idxs))
        else:
            n_init = n0.copy()
        
        num_iter = 0
        while True:
            ntrial = fsolve(growth_rates_solve,n_init,args=(extant_idxs))
            remove_idxs = np.nonzero(ntrial<=0)[0]
            extinct_idxs = np.append(extinct_idxs,extant_idxs[remove_idxs])
            extant_idxs = np.delete(extant_idxs,remove_idxs)
            n_init = np.delete(n_init,remove_idxs)
            extinct_biases = growth_rates(ntrial[ntrial>0],extant_idxs)[extinct_idxs]

            if np.all(ntrial>0) and (np.all(extinct_biases<0) or num_iter>=max_iter):
                if num_iter>=max_iter:
                    self.max_iter_epochs.append(ii)
                if self.check_stability:
                    if not self.is_stable(ntrial,G[extant_idxs,:],F[extant_idxs,:]):
                        self.unstable_epochs.append(ii)
                return ntrial,extant_idxs
            
            if num_iter<max_iter:
                extant_idxs = np.append(extant_idxs,extinct_idxs[extinct_biases>0])
                if n0 is None:
                    add_strains = [1]*np.sum(extinct_biases>0)
                else:
                    add_strains = n0[extinct_idxs[extinct_biases>0]]
                n_init = np.append(n_init,add_strains)
                extinct_idxs = np.delete(extinct_idxs,np.nonzero(extinct_biases>0)[0])
                num_iter += 1

    # check stability of fixed point
    def is_stable(self,n_sol,G,F):
        K_tilde = self.Kvec/(self.omega+np.dot(F.T,n_sol))**2
        jacobian = -np.linalg.multi_dot((np.diag(n_sol),G,np.diag(K_tilde),F.T))
        stability_eigvals = np.linalg.eigvals(jacobian)
        return np.all(np.real(stability_eigvals)<0)

    # main method to run evolution
    def run_resource_evo(self):
        mu = self.mu
        m = self.m
        rho = self.mut_corr
        sigma = self.sigma
        kappa = self.kappa
        D = self.D
        omega = self.omega

        for epoch_idx in range(self.num_epochs):
            if epoch_idx%1000==0:
                print(epoch_idx)
            if epoch_idx==0:
                n0=None

            n_sol,extant_idxs = self.find_fp_ext(epoch_idx,n0=n0)
            self.idx_list = self.idx_list[extant_idxs]

            if epoch_idx>0:
                self.coex_list.append(par_idx in extant_idxs)

            self.G = self.G[extant_idxs,:]
            self.F = self.F[extant_idxs,:]

            Rvec = self.Kvec/(omega+np.dot(self.F.T,n_sol))
            assert(np.all(Rvec>=0)) # make sure that resource abundances stay positive
            inv_bias = 0

            # generate mutant
            par_idx = np.random.choice(len(n_sol),p=n_sol/sum(n_sol))
            g_par = self.G[par_idx,:] - mu
            quant = (m-mu*np.sum(Rvec)-rho*np.dot(g_par,Rvec))/np.sqrt(1-rho**2)
            random_part = sample_truncated(sigma*Rvec,quant)
            g_mut = rho*g_par + sigma*np.sqrt(1-rho**2)*random_part

            inv_bias = np.dot(g_mut+mu,Rvec) - m
            assert(inv_bias>0)

            f_par = self.F[par_idx,:] - mu
            f_mut = rho*f_par + sigma*np.sqrt(1-rho**2)*(kappa*random_part + np.sqrt(1-kappa**2)*np.random.randn(D))

            n0 = n_sol
            n0 = np.append(n0,[1])

            if epoch_idx%self.sample_rate==0:
                g_mat = self.G - np.mean(self.G)
                f_mat = self.F - np.mean(self.F)
                self.GK_list.append(np.dot(self.G,self.Kvec))
                self.Fn_list.append(np.dot(self.F.T,n_sol))
                self.R_list.append(Rvec.copy())
                self.L_list.append(len(n_sol))
                self.n_list.append(n_sol.copy())
                self.strain_idx_list.append(self.idx_list.copy())
                self.epoch_list.append(epoch_idx)
                self.inv_prob.append(norm.logsf(quant/(sigma*np.linalg.norm(Rvec))))

            # insert mutant
            self.G = np.vstack([self.G,g_mut+mu])
            self.F = np.vstack([self.F,f_mut+mu])
            self.idx_list = np.append(self.idx_list,[self.count])
            self.count += 1

            if epoch_idx%self.epoch_save_num==0:
                self.save_state()

        self.save_state()

    def save_state(self):
        data = vars(self).copy()
        np.savez(self.filename,data=data)