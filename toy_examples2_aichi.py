"""
This example is to present the situation of detecting "where the bias is" 
by checking the residual distribution of each measurement

Ans: The one with central chi-square distribution reveals the location of bias
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, ncx2

x = np.array([[1], [2]])
H = np.array([[1, 0], [0, 1], [1, 1], [2, 3], [-1, 1]])
mu = np.array([[0], [0], [0], [5], [0]]) # unknown in real 
m = H.shape[0]
n = H.shape[1]
df = m - n  
dfi = m - n -1 # Hi.shape[0] - Hi.shape[1]
num_run = 2000

e_matrix = np.eye(m)
z_array = np.zeros((H.shape[0], num_run))
nc_array = np.zeros((H.shape[0]))
color_array = ["skyblue", "lightcoral", "lightgreen", "gold", "plum", "orange", "mediumseagreen"]

for i in range(num_run):
    epsilon = np.random.randn(m,1)
    yi = H @ x + mu + epsilon # generate observation which content bias

    for j in range (H.shape[0]):
        Hj = np.hstack((H, e_matrix[:,j].reshape(-1, 1)))
        Sj = np.linalg.pinv(Hj)
        x_est = Sj @ yi
        z_est = yi - Hj @ x_est
        z_array[j, i] = z_est.T @ z_est
        nc_array[j] = mu.T @ (np.eye(Hj.shape[0])- Hj @ Sj) @ mu # unknown in real


# use one column of z_array to determine which measurement is biased
# bc in real case we always have one set of measurement
pdf_values = chi2.pdf(z_array[:,1],dfi)
max_index = np.argmax(pdf_values)
print(f"The {max_index+1} is the biased measurement")

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

