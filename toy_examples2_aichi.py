"""
This example is to present the situation of detecting "where the bias is" 
by checking the residual distribution of each measurement

Ans: The one with central chi-square distribution reveals the location of bias
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, ncx2

def generate_Hi(num_place, H):
    if num_place >= H.shape[0]: 
        raise ValueError("num_place exceeds matrix row dimensions.")
    else:
        e = np.zeros((H.shape[0], 1))
        e[num_place] = 1
        Hi = np.hstack((H, e))
        return Hi, e

x = np.array([[1], [2]])
H = np.array([[1, 0], [0, 1], [1, 1], [2, 3], [-1, 1]])
mu = np.array([[3], [0], [0], [0], [0]]) # not adding in H0 case
m = H.shape[0]
n = H.shape[1]
df = m - n  
S = np.linalg.pinv(H)

num_run = 5000

z_array = np.zeros((H.shape[0], num_run))
nc_array = np.zeros((H.shape[0]))
color_array = ["skyblue", "lightcoral", "lightgreen", "gold", "plum", "orange", "mediumseagreen"]

for i in range(H.shape[0]):

    # set new form 
    Hi, e = generate_Hi(i, H)
    mui = abs(np.max(mu)) * e
    xi = np.vstack((x, abs(np.max(mu))))
    Si = np.linalg.pinv(Hi)
    dfi = Hi.shape[0] - Hi.shape[1]
    nc_array[i] = mu.T @ (np.eye(Hi.shape[0])- Hi @ Si) @ mu

    for j in range (num_run):
        epsilon = np.random.randn(m,1)
        y = H @ x + mu + epsilon
        x_est = Si @ y
        z_est = y - Hi @ x_est
        z_array[i, j] = z_est.T @ z_est


# plot the distribution and the theoretical distribution
x_vals = np.linspace(0, np.max(z_array), 500)
for i in range(H.shape[0]):

    plt.hist(z_array[i, :], bins=30, density=True, alpha=0.5, color= color_array[i], edgecolor='#888888', label='Empirical distribution of z (central)')
    ncx2_pdf = ncx2.pdf(x_vals, dfi, float(nc_array[i])) 
    plt.plot(x_vals, ncx2_pdf, color=color_array[i], linestyle='--', label=f'Non-central Chi-squared PDF (df={df}, λ={float(nc_array[i]):.2f})')


# figure setting
plt.xlabel("Value of z")
plt.ylabel("Probability Density")
plt.title("Empirical Distribution of z and Chi-squared PDF")
plt.legend()
plt.show()

