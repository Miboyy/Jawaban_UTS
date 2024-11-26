# prompt: Tuliskan program sederhana menggunakan Python untuk melakukan Regresi Linear dengan library NumPy dan scikit-learn. Data yang digunakan adalah:
# text
# Copy
# X = [1, 2, 3, 4, 5]
# Y = [3, 6, 9, 12, 15]
# Lakukan prediksi untuk nilai Y jika x = 6

import numpy as np
from sklearn.linear_model import LinearRegression

# Data
X = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
Y = np.array([3, 6, 9, 12, 15])

# Membuat model regresi linear
model = LinearRegression()

# Melatih model dengan data
model.fit(X, Y)

# Melakukan prediksi untuk x = 6
x_new = np.array([6]).reshape((-1, 1))
y_pred = model.predict(x_new)

# Menampilkan hasil prediksi
print(f"Prediksi nilai Y untuk x = 6: {y_pred[0]}")