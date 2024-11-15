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
mu_array = 5 * np.eye(m) # unknown in real 
mu = np.array([[0], [5], [0], [0], [0]]) # unknown in real 
df = m - n  
dfi = m - n -1 # Hi.shape[0] - Hi.shape[1]
num_run = 2

color_array = ["skyblue", "lightcoral", "lightgreen", "gold", "plum", "orange", "mediumseagreen"]
z_array = np.zeros((H.shape[0], num_run))
nc_array = np.zeros((H.shape[0]))
conf_matrix = np. zeros([5,5])
e_matrix = np.eye(m)

for k in range(mu_array.shape[0]):
    mu = mu_array[:,k].reshape(5,1)
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
        conf_matrix[k, max_index] += 1

conf_matrix = conf_matrix/np.sum(conf_matrix)
# figure setting
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=[f"Error {i}" for i in range(m)])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

