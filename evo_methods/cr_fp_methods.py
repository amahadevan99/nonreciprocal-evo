import numpy as np
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

'''
############################################################################
Here, instead of integrating the dynamics we iteratively try to find the fixed point by eliminating
strains whose abundance is negative by naively inverting the matrix. This seems to converge the majority of the time.
We find that this approach recapitulates the phenomenology of simulating the full dynamics.
############################################################################
'''

class CREvo:
    def __init__(self,filename,K,D,kappa,num_epochs,sample_rate=1,sig_s=0,mut_corr=0,max_iter=10,supply_vec=None,
        epochs_to_save_V=[],epoch_save_num=10000,seed=None,check_stability=True):
        if seed is None:
            self.seed = np.random.randint(4294967296)
        else:
            self.seed = seed
        np.random.seed(seed=self.seed)
        self.filename = filename # file to save data as npz
        self.K = K # initial number of strains
        self.D = D # number of resources
        self.mut_corr = mut_corr # parent mutant correlation
        self.kappa = kappa # symmetry parameter
        self.sample_rate = sample_rate # parameter to subsample data for long runs
        self.num_epochs = num_epochs # length of simulations (in evolutionary epochs)
        self.epochs_to_save_V = epochs_to_save_V # list of epochs in which to save interaction matrix
        self.epoch_save_num = epoch_save_num # how often to save simulation data during run
        self.parent_dict = {} # dictionary recording ancestry
        self.max_iter = max_iter # maximum number of iterations to add back invadable strains when finding fixed point
        self.sig_s = sig_s # width of gaussian distribution of general fitnesses
        self.check_stability = check_stability # whether or not to check stability of fixed points
        if supply_vec is None:
            self.supply_vec = np.zeros(self.D)
        else:
            self.supply_vec = supply_vec # resource supply vector

        self.V = self.gen_V() # interaction matrix
        self.idx_list = np.arange(K) # list of extant strains in community, each with a unique identifier
        self.count = K # identifier of next strain to be added
        self.s_list = sig_s*np.array([np.random.randn() for i in range(K)]) # general fitnesses
        self.supply_s = np.dot(self.A,self.supply_vec) # general fitnesses coming from resource supply

        self.L_list = [] # diversity over time
        self.ups_list = [] # upsilon over time
        self.n_list = [] # strain abundances over time
        self.strain_idx_list = [] # strain identifiers over time
        self.s_dist_list = [] # intrinsic general fitnesses over time
        self.supply_s_list = [] # general fitnesses (from resource supply) over time
        self.A_list = [] # list of saved growth matrices
        self.B_list = [] # list of saved consumption matrices
        self.epoch_list = [] # epochs in which data were saved
        self.R_list = [] # resource abundances over time
        self.max_iter_epochs = [] # epochs in which uninvadable fixed point not found
        self.unstable_epochs = [] # epochs in which fixed point is unstable
        self.coex_bools = [] # whether the mutant coexists with its parent or not
        self.inv_bias_list = [] # growth rate of invader at low abundance
        self.inv_prob_list = [] # log invasion probability over time


    def gen_V(self):
        kappa = self.kappa
        K = self.K
        D = self.D
        self.A = np.random.randn(K,D)
        self.B = kappa*self.A + np.sqrt(1-kappa**2)*np.random.randn(K,D)
        V = -np.dot(self.A,self.B.T)
        return V

    # gets resource abundances from supply, consumption matrices, and species abundances
    def get_R_vec(self,nlist):
        return self.supply_vec - np.dot(self.B.T,nlist)

    # adds a single independent strain, conditional on that strain having positive bias
    # returns this invader eigenvalue (bias), parent identifier, and current invasion probability
    def add_strain(self,n_end,upsilon,par_idx=None):
        rho = self.mut_corr
        eps = np.sqrt(1-rho**2)
        K = np.shape(self.V)[0]
        new_V = np.zeros((K+1,K+1))
        new_V[:K,:K] = self.V
        R = self.get_R_vec(n_end)

        if par_idx is None:
            par_idx = np.random.choice(K,p=n_end)

        par_A = self.A[par_idx]
        par_s = self.s_list[par_idx]
        sig_s = self.sig_s
        quant = (1-rho)*upsilon/eps

        # no general fitnesses
        if sig_s==0:
            random_part = sample_truncated(R,quant)
            mut_A = rho*par_A + eps*random_part
            inv_bias = np.dot(mut_A,R)-upsilon
            inv_prob = norm.logsf(quant/np.linalg.norm(R))
            new_s = 0

        # include general fitnesses
        elif sig_s>0:
            R_samp = np.append(R,sig_s)
            r_part = sample_truncated(R_samp,quant)
            random_part = r_part[:-1]
            new_s = rho*par_s + eps*sig_s*r_part[-1]
            mut_A = rho*par_A + eps*random_part
            inv_bias = np.dot(mut_A,R)-upsilon+new_s
            inv_prob = norm.logsf(quant/np.linalg.norm(R_samp))
        assert inv_bias>0

        kappa = self.kappa
        par_B = self.B[par_idx]
        mut_B = rho*par_B + eps*(kappa*random_part+np.sqrt(1-kappa**2)*np.random.randn(self.D))

        new_V[K,:K] = -np.dot(self.B,mut_A)
        new_V[:K,K] = -np.dot(self.A,mut_B)
        new_V[K,K] = -np.dot(mut_A,mut_B)

        self.A = np.vstack((self.A,mut_A))
        self.B = np.vstack((self.B,mut_B))
        
        self.V = new_V
        self.s_list = np.append(self.s_list,new_s)
        self.supply_s = np.append(self.supply_s,np.dot(mut_A,self.supply_vec))
        self.parent_dict[self.count] = self.idx_list[par_idx]
        return inv_bias,self.idx_list[par_idx],inv_prob

    # get fixed point given interaction matrix and some subset of surviving strains
    def get_fp_abund_ups(self,V,extant_idxs):
        assert len(extant_idxs)==len(set(extant_idxs))
        K = len(extant_idxs)
        Vsub = V[extant_idxs,:][:,extant_idxs]
        n = np.linalg.solve(Vsub,np.ones(K))
        n /= sum(n)
        return n, np.dot(n,np.dot(Vsub,n))
        

    # tries to converge on an uninvadable and feasible fixed points. If number of iterations exceeded,
    # settles for feasibility 
    def remove_extinct(self,ii):
        extant_idxs = np.arange(np.shape(self.V)[0]) # indices of V that remain
        extinct_idxs = np.array([],dtype=int)
        iter_count = 0
        max_iter = self.max_iter # number of iterations for which we try to add back invadable strains
        # if this is exceeded, then settle for a feasible fixed point
        s_eff = self.s_list+self.supply_s

        while True:
            ntrial,ups = self.get_fp_abund_ups((self.V.T + s_eff).T,extant_idxs)
            remove_idxs = np.nonzero(ntrial<0)[0]
            extinct_idxs = np.append(extinct_idxs,extant_idxs[remove_idxs])
            extant_idxs = np.delete(extant_idxs,remove_idxs)
            extinct_biases = np.dot((self.V.T + s_eff).T[extinct_idxs,:][:,extant_idxs],ntrial[ntrial>0]) - ups

            if np.all(ntrial>0) and (np.all(extinct_biases<0) or iter_count>=max_iter):
                if not np.all(extinct_biases<0):
                    self.max_iter_epochs.append(ii)
                
                if self.check_stability:
                    if not self.is_stable((self.V[extant_idxs,:][:,extant_idxs].T + s_eff[extant_idxs]).T):
                        self.unstable_epochs.append(ii)
                
                self.s_list = self.s_list[extant_idxs]
                self.supply_s = self.supply_s[extant_idxs]
                self.V = self.V[extant_idxs,:][:,extant_idxs]
                self.A = self.A[extant_idxs,:]
                self.B = self.B[extant_idxs,:]
                self.idx_list = self.idx_list[extant_idxs]

                return ntrial,ups

            # tries to add back strains with positive bias
            if iter_count < max_iter:
                extant_idxs = np.append(extant_idxs,extinct_idxs[extinct_biases>0])
                extinct_idxs = np.delete(extinct_idxs,np.nonzero(extinct_biases>0)[0])
                iter_count += 1

    # check stability of fixed point from V matrix
    def is_stable(self,V):
        K = np.shape(V)[0]
        n,ups = self.get_fp_abund_ups(V,np.arange(K))
        A = np.dot(np.diag(n),V-np.outer(np.dot(V+V.T,n),np.ones(K)).T) # stability matrix
        eigs,eigvecs = np.linalg.eig(A)
        return np.sum(np.real(eigs)>0)<=1

    # main method to run evolution
    def run_evo(self):
        for i in range(self.num_epochs):
            nu,ups = self.remove_extinct(i)

            if i%self.sample_rate==0:
                self.L_list.append(np.shape(self.V)[0])
                self.ups_list.append(ups)
                self.n_list.append(nu)
                self.s_dist_list.append(self.s_list)
                self.supply_s_list.append(self.supply_s)
                self.strain_idx_list.append(self.idx_list)
                self.epoch_list.append(i)
                self.R_list.append(self.get_R_vec(nu))
                if i>0:
                    self.coex_bools.append(par_strain in self.idx_list)
                    self.inv_bias_list.append(inv_bias)
                    self.inv_prob_list.append(inv_prob)
                self.recent_A = self.A.copy()
                self.recent_B = self.B.copy()
            if i in self.epochs_to_save_V:
                self.A_list.append(self.A.copy())
                self.B_list.append(self.B.copy())
            if i%1000==0:
                print(i,self.V.shape[0],flush=True)
            
            inv_bias,par_strain,inv_prob = self.add_strain(nu,ups)
            self.idx_list = np.append(self.idx_list,self.count)
            self.count += 1

            if i%self.epoch_save_num==0:
                self.save_state()

        self.save_state()

    def save_state(self):
        data = vars(self).copy()
        np.savez(self.filename,data=data)

