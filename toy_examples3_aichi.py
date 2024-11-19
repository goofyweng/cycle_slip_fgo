"""
Set the bias at different place, and form confusion matrix
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, ncx2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

x = np.array([[1], [2]])
H = np.array([[1, 0], [0, 1], [1, 1], [2, 3], [-1, 1]])
m = H.shape[0]
n = H.shape[1]

# create the fault
mu_array = 7 * np.eye(m) # unknown in real 
zero_row = np.zeros((mu_array.shape[0], 1))
mu_array = np.hstack((zero_row, mu_array))

dfi = m - n -1 # Hi.shape[0] - Hi.shape[1]
num_run = 50
T = chi2.ppf(1 - 0.001, dfi)

z_array = np.zeros((H.shape[0], num_run))
conf_matrix = np. zeros([mu_array.shape[1],mu_array.shape[1]])
e_matrix = np.eye(m)

for k in range(mu_array.shape[1]):
    mu = mu_array[:,k].reshape(m,1)
    for i in range(num_run):
        epsilon = np.random.randn(m,1)
        yi = H @ x + mu + epsilon # generate observation which content bias

        for j in range (H.shape[0]):
            Hj = np.hstack((H, e_matrix[:,j].reshape(-1, 1)))
            Sj = np.linalg.pinv(Hj)
            x_est = Sj @ yi
            z_est = yi - Hj @ x_est
            z_array[j, i] = z_est.T @ z_est # 5*2000

        pdf_values = chi2.pdf(z_array[:,i],dfi)
        max_index = np.argmax(pdf_values)

        if pdf_values[max_index] > T:
            conf_matrix[k, max_index] += 1
        else:
            conf_matrix[k, 0] += 1

conf_matrix = conf_matrix/np.sum(conf_matrix)
# figure setting
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=[f"Error {i}" for i in range(m+1)])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

