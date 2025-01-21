"""
Detection + identification: form confusion matrix include the case 
when there's no fault
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def generate_fault(H):
    m = H.shape[0]
    fault = np.random.choice([0, 1]) 
    location = np.random.randint(0, m) if fault == 1 else None
    return fault, location

def fault_detection(y, H, T):
    S = np.linalg.pinv(H)
    x_est = S @ y
    z_est = y - H @ x_est
    z = z_est.T @ z_est
    return 0 if z < T else 1

def fault_identification(y, H):
    m = H.shape[0]
    n = H.shape[1]
    e_test_matrix = np.eye(m)
    z_array = np.zeros((m, 1))
    for i in range(m):
        Hj = np.hstack((H, e_test_matrix[:, i].reshape(-1, 1)))
        Sj = np.linalg.pinv(Hj)
        x_est = Sj @ y
        z_est = y - Hj @ x_est
        z_array[i] = z_est.T @ z_est

    pdf_values = chi2.pdf(z_array, m-n-1)
    max_index = np.argmax(pdf_values)
    return max_index

if __name__ == "__main__":
    
    x = np.array([[1], [2]])
    H = np.array([[1, 0], [0, 1], [1, 1], [1, 3], [-1, 1]])
    m = H.shape[0]
    n = H.shape[1]
    mu = 5
    num_run = 2000

    pfa = 0.001
    T = chi2.ppf(1 - pfa, m - n)
    predict_value = np.zeros((num_run, 1))
    true_value = np.zeros((num_run, 1))

    for i in range(num_run):
        # fault generation
        e = np.zeros((m, 1))
        fault, location = generate_fault(H)
        if fault == 1 and location is not None:
            e[location] = 1
            true_value[i] = location + 1
        else:
            true_value[i] = 0

        # measurement generation
        epsilon = np.random.randn(m,1)
        y = H @ x + mu * e + epsilon

        # fault detection
        fault_bool = fault_detection(y, H, T)

        if fault_bool:
            # fault identification
            max_index = fault_identification(y, H)
            predict_value[i] = max_index + 1
        else:
            predict_value[i] = 0

    # draw
    cm = confusion_matrix(true_value, predict_value)
    cm = cm / np.sum(cm) * 100  
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    disp.ax_.set_xticklabels([f"$e_{i}$" for i in range(m + 1)])
    disp.ax_.set_yticklabels([f"$e_{i}$" for i in range(m + 1)])
    disp.ax_.set_xlabel("Predicted error, $e_i$")
    disp.ax_.set_ylabel("True error, $e_j$")
    disp.ax_.set_title(f"No fault: $e_0$;  Fault: $e_i, i \\in [1,...,{m}]$")
    plt.show()
