"""
This example is to present the situation of "detecting bias" 
by observing the residual distribution 

Ans: The one with non-central chi-square distribution has bias
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, ncx2

x = np.array([[1], [2]])
H = np.array([[1, 0], [0, 1], [1, 1], [2, 3], [-1, 1]])
mu = np.array([[3], [0], [0], [0], [0]]) # not adding in H0 case
m = H.shape[0]
n = H.shape[1]
df = m - n  
S = np.linalg.pinv(H)

num_run = 5000
z_array0 = np.zeros((num_run,))
z_array1 = np.zeros((num_run,))
nc0 = 0 # centrality
nc1 = mu.T @ (np.eye(m)- H @ S) @ mu

for i in range(num_run):
    epsilon = np.random.randn(m,1) # It's important to use normal distribution, and 
    y0 = H @ x + epsilon
    x_est0 = S @ y0
    z_est0 = y0 - H @ x_est0
    z_array0[i] = z_est0.T @ z_est0

    y1 = H @ x + epsilon + mu
    x_est1 = S @ y1
    z_est1 = y1 - H @ x_est1
    z_array1[i] = z_est1.T @ z_est1
 
# plot the distribution of z
plt.hist(z_array0, bins=30, density=True, alpha=0.5, color='skyblue', edgecolor='#888888', label='Empirical distribution of z (central)')
plt.hist(z_array1, bins=30, density=True, alpha=0.5, color='lightcoral', edgecolor='#888888', label='Empirical distribution of z (non-central)')

# plot the theoretical distribution
x_vals = np.linspace(0, max(np.max(z_array0), np.max(z_array1)), 500)
chi2_pdf = chi2.pdf(x_vals, df)
ncx2_pdf = ncx2.pdf(x_vals, df, float(nc1)) 
plt.plot(x_vals, chi2_pdf, color='blue', linestyle='--', label=f'Chi-squared PDF (df={df})')
plt.plot(x_vals, ncx2_pdf, color='red', linestyle='--', label=f'Non-central Chi-squared PDF (df={df}, λ={float(nc1):.2f})')

# figure setting
plt.xlabel("Value of z")
plt.ylabel("Probability Density")
plt.title("Empirical Distribution of z and Chi-squared PDF")
plt.legend()
plt.show()
