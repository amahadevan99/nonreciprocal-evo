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


##############################################################################
# Here, instead of integrating the dynamics we iteratively try to find the fixed point by eliminating
# strains whose abundance is negative by naively inverting the matrix. This seems to converge.
# We find that this approach recapitulates the phenomenology of simulating the full dynamics.
##############################################################################

class LVEvo:

    def __init__(self,filename,K,Q,gamma,num_epochs,sample_rate=1,sig_s=0,epochs_to_save_V=[],epoch_save_num=10000,
        seed=None,max_iter=10,check_stability=True):
        if seed is None:
            self.seed = np.random.randint(4294967296)
        else:
            self.seed = seed
        np.random.seed(seed=self.seed)
        self.filename = filename # file to save data as npz
        self.K = K # initial number of strains
        self.Q = Q # negative diagonal of matrix
        self.gamma = gamma # symmetry parameter
        self.sample_rate = sample_rate # parameter to subsample data for long runs
        self.num_epochs = num_epochs # length of simulations (in evolutionary epochs)
        self.epochs_to_save_V = epochs_to_save_V # list of epochs in which to save interaction matrix
        self.epoch_save_num = epoch_save_num # how often to save simulation data during run
        self.max_iter = max_iter # maximum number of iterations to add back invadable strains when finding fixed point
        self.check_stability = check_stability # whether or not to check stability of fixed points
        self.sig_s = sig_s # width of gaussian distribution of general fitnesses
        self.s_list = sig_s*np.random.randn(K) # list of extant fitnesses in population

        self.V = self.gen_V() # interaction matrix
        self.idx_list = np.arange(K) # list of extant strains in community, each with a unique identifier
        self.count = K # identifier of next strain to be added

        self.L_list = [] # diversity over time
        self.ups_list = [] # upsilon over time
        self.n_list = [] # strain abundances at fixed point, over time
        self.strain_idx_list = [] # extant strain identifiers over time
        self.V_list = [] # saved interaction matrices
        self.epoch_list = [] # epochs in which data were saved
        self.max_iter_epochs = [] # epochs in which uninvadable fixed point not found
        self.unstable_epochs = [] # epochs in which fixed point is unstable
        self.inv_prob_list = [] # log invasion probability over time
        self.s_dist_list = [] # extant strain fitnesses over time

    def gen_V(self):
        V = np.random.randn(self.K,self.K)
        if self.gamma==0:
            a = 0
        else:
            a = (1-np.sqrt(1-self.gamma**2))/self.gamma
        V = (V + a*V.T)/np.sqrt(1+a**2)
        V -= np.diag(np.ones(self.K)*self.Q)
        return V

    # adds a new row and column to the interaction matrix and calculate the invasion probability of the invader
    def add_strain(self,n_end,upsilon):
        K = np.shape(self.V)[0]
        sig_s = self.sig_s

        new_V = np.zeros((K+1,K+1))
        new_V[:K,:K] = self.V
        
        if sig_s==0:
            row = random_part = sample_truncated(n_end,upsilon)
            inv_bias = np.dot(row,n_end)-upsilon
            inv_prob = norm.logsf(upsilon/np.linalg.norm(n_end))
            new_s = 0
        else:
            nu_samp = np.append(n_end,sig_s)
            r_part = sample_truncated(nu_samp,upsilon)
            new_s = sig_s*r_part[-1]
            row = random_part = r_part[:-1]
            inv_bias = np.dot(row,n_end)-upsilon + new_s
            inv_prob = norm.logsf(upsilon/np.linalg.norm(nu_samp))

        assert inv_bias>0
        col = self.gamma*random_part+np.sqrt(1-self.gamma**2)*np.random.randn(K)
        new_V[K,:K] = row
        new_V[:K,K] = col

        new_V[K,K] = np.sqrt(1+self.gamma)*np.random.randn() - self.Q
        self.s_list = np.append(self.s_list,new_s)
        self.V = new_V
        return inv_prob

    # given an interaction matrix, computes the fixed point corresponding to a specified submatrix
    def get_fp_abund_ups(self,V,extant_idxs):
        K = len(extant_idxs)
        Vsub = V[extant_idxs,:][:,extant_idxs]
        n = np.linalg.solve(Vsub,np.ones(K))
        n /= sum(n)
        return n, np.dot(n,np.dot(Vsub,n))

    # tries to converge on an uninvadable and feasible fixed points. If number of iterations exceeded, settles for feasibility 
    def remove_extinct(self,ii):
        extant_idxs = np.arange(np.shape(self.V)[0]) # indices of V for which to calculate fixed point
        extinct_idxs = np.array([],dtype=int)
        iter_count = 0
        max_iter = self.max_iter # number of iterations for which we try to add back invadable strains

        while True:
            ntrial,ups = self.get_fp_abund_ups((self.V.T + self.s_list).T,extant_idxs)
            remove_idxs = np.nonzero(ntrial<0)[0]
            extinct_idxs = np.append(extinct_idxs,extant_idxs[remove_idxs])
            extant_idxs = np.delete(extant_idxs,remove_idxs)
            extinct_biases = np.dot((self.V.T + self.s_list).T[extinct_idxs,:][:,extant_idxs],ntrial[ntrial>0]) - ups

            # checks for uninvadable fixed point
            if np.all(ntrial>0) and (np.all(extinct_biases<0) or iter_count>=max_iter):
                if not np.all(extinct_biases<0):
                    self.max_iter_epochs.append(ii)
                if self.check_stability:
                    if not self.is_stable(self.V[extant_idxs,:][:,extant_idxs]+self.s_list[extant_idxs].T):
                        self.unstable_epochs.append(ii)

                self.s_list = self.s_list[extant_idxs]
                self.V = self.V[extant_idxs,:][:,extant_idxs]
                self.idx_list = self.idx_list[extant_idxs]
                return ntrial,ups

            # tries to add back strains with positive bias, if sufficiently few iterations run already
            if iter_count < max_iter:
                extant_idxs = np.append(extant_idxs,extinct_idxs[extinct_biases>0])
                extinct_idxs = np.delete(extinct_idxs,np.nonzero(extinct_biases>0)[0])
                iter_count += 1

    # save data to file
    def save_state(self):
        data = vars(self).copy()
        np.savez(self.filename,data=data)

    # check stability of fixed point from V matrix: allow one unstable eigenvector due to the upsilon constraint
    def is_stable(self,V):
        K = np.shape(V)[0]
        n,ups = self.get_fp_abund_ups(V,np.arange(K))
        A = np.dot(np.diag(n),V-np.outer(np.dot(V+V.T,n),np.ones(K)).T) # stability matrix
        eigs,eigvecs = np.linalg.eig(A)
        return np.sum(np.real(eigs)>0)<=1

    # main method to run simulation
    def run_evo(self):
        for i in range(self.num_epochs):
            nu,ups = self.remove_extinct(i)

            if i%self.sample_rate==0:
                self.L_list.append(np.shape(self.V)[0])
                self.ups_list.append(ups)
                self.n_list.append(nu)
                self.strain_idx_list.append(self.idx_list)
                self.s_dist_list.append(self.s_list)
                self.epoch_list.append(i)
                if i>0:
                    self.inv_prob_list.append(inv_prob)
                    
            if i in self.epochs_to_save_V:
                self.V_list.append(self.V.copy())

            # prints out progress log
            if i%1000==0:
                print(i,self.V.shape[0],flush=True)
            
            inv_prob = self.add_strain(nu,ups)
            self.idx_list = np.append(self.idx_list,self.count)
            self.count += 1

            if i%self.epoch_save_num==0:
                self.save_state()

        self.save_state()
