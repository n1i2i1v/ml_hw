from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

A = np.matrix(load_digits().data)

def u_d_v(A, index):
    at_a = np.dot(A.T, A)
    S = np.diag(np.sqrt(eigh(at_a)[0])[::-1])
    for i in range(len(S)):
        if i ==63:
            S[i] += 10e-10
    S_inv = np.linalg.inv(S)
    V = ((eigh(at_a)[1])[::-1])
    V_T = V.T
    U = A * V * S_inv

    #calculate the new one
    A_new = U[:,0:index] * S[0:index:,0:index] * V_T[0:index:,:]
    
    return A_new

for i in [2, 4, 8, 16, 32, 64]:
    plt.figure(figsize=(6,6))
    A_modified = u_d_v(A, i)
    plt.imshow(A_modified[0].reshape((8, 8)), cmap="gray_r")
    plt.show()