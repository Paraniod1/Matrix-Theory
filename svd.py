import numpy as np
import cv2
import matplotlib.pyplot as plt

# 读取图像并进行灰度化
image = cv2.imread(r'D:\Data\pythonProject\svd.jpg')  # 读取图像
A_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度图

# 获取灰度图像的尺寸
m, n = A_gray.shape
print(f"Image dimensions: {m}x{n}")

# 设置不同的 k 值（保留的奇异值个数）
k1 = 20
k2 = 40
k3 = 60
k4 = 80

# 计算压缩比 ρ = mn / (k * (m + n + 1))
rho1 = m * n / (k1 * (m + n + 1))
rho2 = m * n / (k2 * (m + n + 1))
rho3 = m * n / (k3 * (m + n + 1))
rho4 = m * n / (k4 * (m + n + 1))

print(f"Compression ratios: {rho1:.4f}, {rho2:.4f}, {rho3:.4f}, {rho4:.4f}")

# 将灰度矩阵转换为浮点型
A_gray1 = np.float64(A_gray)

# 进行奇异值分解 (SVD)
U, S, Vt = np.linalg.svd(A_gray1, full_matrices=False)

# 将奇异值矩阵 S 转换为对角矩阵
S1 = np.copy(S)
S2 = np.copy(S)
S3 = np.copy(S)
S4 = np.copy(S)

# 截取前 k 个奇异值并将后面的奇异值设为 0
S1[k1:] = 0
S2[k2:] = 0
S3[k3:] = 0
S4[k4:] = 0

# 重新构建压缩后的图像
A1_compressed = np.dot(U, np.dot(np.diag(S1), Vt))
A2_compressed = np.dot(U, np.dot(np.diag(S2), Vt))
A3_compressed = np.dot(U, np.dot(np.diag(S3), Vt))
A4_compressed = np.dot(U, np.dot(np.diag(S4), Vt))

# 转换为 uint8 类型，适合显示
A1_compressed = np.uint8(np.clip(A1_compressed, 0, 255))
A2_compressed = np.uint8(np.clip(A2_compressed, 0, 255))
A3_compressed = np.uint8(np.clip(A3_compressed, 0, 255))
A4_compressed = np.uint8(np.clip(A4_compressed, 0, 255))

# 显示压缩后的图像
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(A1_compressed, cmap='gray')
plt.title(f"Compressed (k={k1})")
plt.subplot(2, 2, 2)
plt.imshow(A2_compressed, cmap='gray')
plt.title(f"Compressed (k={k2})")
plt.subplot(2, 2, 3)
plt.imshow(A3_compressed, cmap='gray')
plt.title(f"Compressed (k={k3})")
plt.subplot(2, 2, 4)
plt.imshow(A4_compressed, cmap='gray')
plt.title(f"Compressed (k={k4})")
plt.tight_layout()
plt.show()