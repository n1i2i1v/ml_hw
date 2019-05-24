import numpy as np
from scipy.stats import norm
from scipy.special import gamma, factorial2
import seaborn as sns
import matplotlib.pyplot as plt

#helper gaussian integral
def gauss_integral(n):
    factor = np.sqrt(np.pi * 2)
    if n % 2 == 0:
        return factor * factorial2(n - 1) / 2
    elif n % 2 == 1:
        return factor * norm.pdf(0) * factorial2(n - 1)
    
#makes the data
def datainator(N, f=0.5, sd=2):
    rand = np.random.RandomState(sd)
    x = rand.randn(N)
    x[int(f * N):] += 1.73
    return x
    
# define the box, gaussian and epanechikov kernels
def gaussian(x, dims=1):
    norm = dims * gauss_integral(dims - 1)
    dist_sq = x ** 2
    return np.exp(-dist_sq / 2) / norm

def epanechnikov(x):
    if abs(x) < 1:
        return 3/4*(1-x**2)
    else:
        return 0

def box(x, dims=1):
    norm = 1
    out = np.zeros_like(x)
    mask = x < 1
    out[mask] = 1 / norm
    return out


#KDE class
class KDE(object):
    def __init__(self, kernel, h = 0.2):
        self.h = h 
        self.kernel = kernel
    
    def fit(self, data):
        self.data = data
    
    def predict(self, x):
        N = len((self.data))
        result = 0
        for datum in self.data:
            y = (datum-x)/self.h
            result += self.kernel(y)
        return 1/N*result
    
    def plot_distirbution(self):
        if self.kernel.__name__ == 'gaussian':
            kernel = 'gau'
        elif self.kernel.__name__ == 'epanechnikov':
            kernel = 'epa'
        else:
            kernel = 'biw'
        plt.figure(figsize=(9, 6))
        plt.subplot(2,1,2)
        sns.kdeplot(data=self.data, bw=0.5, kernel = kernel)
        plt.title(self.kernel.__name__ + ' sns func predict')
        
        plt.figure(figsize=(9, 6))
        plt.subplot(2,1,1)
        plt.title(self.kernel.__name__ + ' self_writter predict')
        predict = np.array([self.predict(datum) for datum in self.data])
        plt.scatter(self.data, predict, marker = '.')

#define the  data and how it looks 
x = datainator(200)
hist = plt.hist(x, bins=30, normed=True)

#tests on data
model_box = KDE(box)
model_box.fit(x)
print(model_box.predict(0))
print(model_box.predict(1.5))
print(model_box.predict(3))
print(model_box.predict(4.5))
model_box.plot_distirbution()

model_ep = KDE(epanechnikov)
model_ep.fit(x)
print(model_ep.predict(0))
print(model_ep.predict(1.5))
print(model_ep.predict(3))
print(model_ep.predict(4.5))
model_ep.plot_distirbution()

model_gaus = KDE(gaussian)
model_gaus.fit(x)
print(model_gaus.predict(0))
print(model_gaus.predict(1.5))
print(model_gaus.predict(3))
print(model_gaus.predict(4.5))
model_gaus.plot_distirbution()