import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 
from scipy.stats import multivariate_normal as mvn

#init data
c3 = np.array(list(zip(np.random.normal(0,0.3,  size=750),
                       np.random.normal(0,0.6,  size=750))))
c2 = np.array(list(zip(np.random.normal(3,0.5,  size=750),
                       np.random.normal(3,0.8,  size=750))))
c1 = np.array(list(zip(np.random.normal(0,0.8,  size=750),
                       np.random.normal(3,0.9,  size=750))))
c0 = np.array(list(zip(np.random.normal(3,0.3,  size=750),
                       np.random.normal(0,0.7,  size=750))))
t = np.hstack((np.ones(750), np.ones(750)*2, np.ones(750)*3, np.zeros(750)))
data = np.vstack((c0, c1, c2, c3))
mask = np.arange(len(data))
np.random.shuffle(mask)
data = data[mask]
t = t[mask]

#K_Means class init
class K_Means(object):
    def __init__(self, n_class, alpha = 0.00018, max_n_iter = 500):
        self.n_class = n_class
        self.alpha = alpha
        self.n_iter = max_n_iter

    def fit(self, data):

        self.means = {}
        for i in range(self.n_class):
            self.means[i] = data[i]
            
        # for every iteration executes the EM algorithm
        for i in range(self.n_iter):
            self.class_all= {}
            for j in range(self.n_class):
                self.class_all[j] = []

            #calculates the r_ik s in each case
            for datum in data:
                #for every point distance from class centers
                distance_all = [np.linalg.norm(datum - self.means[mean]) 
                                for mean in self.means]
                #classifies data
                class_i = distance_all.index(min(distance_all))
                self.class_all[class_i].append(datum)

            #saves the old means as an exit strategy 
            latest = dict(self.means)

            #new means are calculated based on already classified classpoints
            for class_i in self.class_all:
                self.means[class_i] = np.average(self.class_all[class_i], axis = 0)

            #stopping rule 
            enough = True
            for mean in self.means:

                old_mean = latest[mean]
                current_mean = self.means[mean]

                if np.sum((current_mean - old_mean)/old_mean * 100.0) > self.alpha:
                    enough = False

            if enough:
                break

    def predict_one(self, data):
        distance_all = [np.linalg.norm(data - self.means[mean]) for mean in self.means]
        class_i = distance_all.index(min(distance_all))
        return class_i
    
    def predict(self, data):
        return [self.predict_one(datum) for datum in data]


#GMM init
class Gaussian_Mix(object):
    def __init__(self, n_class, n_iter = 50):
        self.C = n_class
        self.n_iter = n_iter
        
    def fit(self, data):
        self.mean, self.sigma, self.pi =  self.init_params(data)
    
        for iteration in range(self.n_iter):  
            self.gamma  = self.e_step(data, self.mean, self.pi, self.sigma)
            self.pi, self.mean, self.sigma = self.m_step(data, self.gamma)                
    
    def init_params(self, data):
        #generates the starting parameters based on kmeans
        n_clusters = self.C
        kmeans = K_Means(n_class= n_clusters)
        kmeans.fit(data)
        prediction = kmeans.predict(data)
        self._initial_means, self._initial_cov, self._initial_pi = self.initiate_mean_cov(data, prediction)
        return(self._initial_means, self._initial_cov, self._initial_pi)
        
    def initiate_mean_cov(self, data, prediction):
        #fits parameters to the given data and prediction
        feat = data.shape[1]
        results= np.unique(prediction)
        self.initial_means = np.zeros((self.C, feat))
        self.initial_cov = np.zeros((self.C, feat, feat))
        self.initial_pi = np.zeros(self.C)
        count=0
        for result in results:
            ifs = np.where(prediction == result) 
            self.initial_pi[count] = len(ifs[0]) / data.shape[0]
            self.initial_means[count,:] = np.mean(data[ifs], axis = 0)
            d_meaned = data[ifs] - self.initial_means[count,:]
            Nk = data[ifs].shape[0]
            self.initial_cov[count,:, :] = np.dot(self.initial_pi[count] * d_meaned.T, d_meaned) / Nk
            count+=1            
        return (self.initial_means, self.initial_cov, self.initial_pi)
            
        
    def e_step(self, data, pi, mu, sigma):
        N = data.shape[0] 
        self.gamma = np.zeros((N, self.C))

        #init params
        self.mean = self.mean if self._initial_means is None else self._initial_means
        self.pi = self.pi if self._initial_pi is None else self._initial_pi
        self.sigma = self.sigma if self._initial_cov is None else self._initial_cov
        
        #get gamma
        for c in range(self.C):
            self.gamma[:,c] = self.pi[c] * mvn.pdf(data, self.mean[c,:], self.sigma[c])

        # normalize to get probability
        gamma_norm = np.sum(self.gamma, axis=1)[:,np.newaxis]
        self.gamma /= gamma_norm

        return self.gamma
    
    
    def m_step(self, data, gamma):
        #cluster amount
        C = self.gamma.shape[1]
        
        #new params
        self.pi = np.mean(self.gamma, axis = 0)
        self.mean = np.dot(self.gamma.T, data) / np.sum(self.gamma, axis = 0)[:,np.newaxis]

        for c in range(C):
            x = data - self.mean[c, :] 

            gamma_diag = np.diag(self.gamma[:,c])
            gamma_diag = np.matrix(gamma_diag)

            sigma_c = x.T * gamma_diag * x
            self.sigma[c,:,:]=(sigma_c) / np.sum(self.gamma, axis = 0)[:,np.newaxis][c]

        return self.pi, self.mean, self.sigma
    
    def predict(self, data):
        results= np.zeros((data.shape[0], self.C))
        for c in range(self.C):
            results[:,c] = self.pi[c] * mvn.pdf(data, self.mean[c,:], self.sigma[c])
        results = results.argmax(1)
        return results


#init models and train
model_means = K_Means(4)
model_means.fit(data)
model_mix = Gaussian_Mix(4)
model_mix.fit(data)

#plots the comparisons 
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.scatter(data[:,0], data[:,1], c =model_mix.predict(data))
plt.title('Predicted GMM')
plt.subplot(2,2,2)
plt.scatter(data[:,0], data[:,1], c =model_means.predict(data))
plt.title('Predicted, K-Means')
plt.subplot(2,2,3)
plt.scatter(data[:,0], data[:,1], c = t)
plt.title('Original')