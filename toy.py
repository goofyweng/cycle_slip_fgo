import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# 原始資料
x = np.array([[1], [2]])
H = np.array([[1, 0], [0, 1], [1, 1]])
m = H.shape[0]
n = H.shape[1]
df = m - n  # 卡方分布的自由度

# 模擬多次 z 的值
num_run = 500
z_array = np.zeros((num_run,))

for i in range(num_run):
    S = np.linalg.pinv(H)
    epsilon = np.random.rand(3, 1)
    y = np.dot(H, x) + epsilon

    x_est = S @ y
    z_est = y - H @ x_est
    z_array[i] = z_est.T @ z_est

# 繪製 z 的經驗分布（直方圖）
plt.hist(z_array, bins=30, density=True, alpha=0.6, color='skyblue', label='Empirical distribution of z')

# 繪製中心卡方分布的 PDF 作為對照
x_vals = np.linspace(0, np.max(z_array), 500)
chi2_pdf = chi2.pdf(x_vals, df)
plt.plot(x_vals, chi2_pdf, color='red', linestyle='--', label=f'Chi-squared PDF (df={df})')

# 圖表標籤和標題
plt.xlabel("Value of z")
plt.ylabel("Probability Density")
plt.title("Empirical Distribution of z and Chi-squared PDF")
plt.legend()
plt.show()
