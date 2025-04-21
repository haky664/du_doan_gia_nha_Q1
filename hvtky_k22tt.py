import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Đọc dữ liệu huấn luyện từ file CSV
train_file_path = 'data.csv'  # Thay 'data.csv' bằng đường dẫn thực tế đến file huấn luyện
try:
    df_train = pd.read_csv(train_file_path)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file huấn luyện tại đường dẫn '{train_file_path}'")
    exit()

# Kiểm tra xem các cột cần thiết có tồn tại trong file huấn luyện không
expected_columns_train = ['Địa chỉ', 'Diện tích (m2)', 'Phòng ngủ', 'WC', 'Giá']
if not all(col in df_train.columns for col in expected_columns_train):
    print(f"Lỗi: File huấn luyện CSV phải chứa các cột: {expected_columns_train}")
    exit()

# Xác định biến độc lập (X_train) và biến phụ thuộc (y_train) cho dữ liệu huấn luyện
X_train_data = df_train[['Địa chỉ', 'Diện tích (m2)', 'Phòng ngủ', 'WC']]
y_train = df_train['Giá']

# Xác định các cột cần mã hóa one-hot và cột số
categorical_features = ['Địa chỉ']
numerical_features = ['Diện tích (m2)', 'Phòng ngủ', 'WC']

# Tạo bộ tiền xử lý cột
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# Áp dụng tiền xử lý cho dữ liệu huấn luyện
X_train_processed = preprocessor.fit_transform(X_train_data)

# Khởi tạo và huấn luyện mô hình hồi quy tuyến tính đa biến
model = LinearRegression()
model.fit(X_train_processed, y_train)

# Đọc dữ liệu kiểm tra từ file CSV
test_file_path = 'aaa.csv'  # Thay 'test.csv' bằng đường dẫn thực tế đến file kiểm tra
try:
    df_test = pd.read_csv(test_file_path)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file kiểm tra tại đường dẫn '{test_file_path}'")
    exit()

# Kiểm tra xem các cột cần thiết có tồn tại trong file kiểm tra không
expected_columns_test = ['Địa chỉ', 'Diện tích (m2)', 'Phòng ngủ', 'WC', 'Giá']
if not all(col in df_test.columns for col in expected_columns_test):
    print(f"Lỗi: File kiểm tra CSV phải chứa các cột: {expected_columns_test}")
    exit()

# Xác định biến độc lập (X_test) và biến phụ thuộc (y_test) cho dữ liệu kiểm tra
X_test_data = df_test[['Địa chỉ', 'Diện tích (m2)', 'Phòng ngủ', 'WC']]
y_test = df_test['Giá']

# Áp dụng tiền xử lý cho dữ liệu kiểm tra (sử dụng bộ tiền xử lý đã fit trên dữ liệu huấn luyện)
X_test_processed = preprocessor.transform(X_test_data)

# Dự đoán trên dữ liệu kiểm tra
y_pred_test = model.predict(X_test_processed)

# Đánh giá hiệu suất của mô hình trên tập kiểm tra
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

# Lấy tên các đặc trưng sau khi tiền xử lý
feature_names_processed = preprocessor.get_feature_names_out(X_train_data.columns)

# In ra hệ số hồi quy
coefficients_df = pd.DataFrame({'Feature': feature_names_processed, 'Coefficient': model.coef_})
print("Hệ số hồi quy:")
print(model.coef_)

# In ra hệ số chặn (intercept)
print(f'\nHệ số chặn (Intercept): {model.intercept_:.2f}')

# In ra các giá trị dự đoán trên tập kiểm tra với định dạng .2f theo mảng
predicted_prices_formatted = [f"{price:.2f}" for price in y_pred_test]
# print("\nGiá dự đoán trên tập kiểm tra:", predicted_prices_formatted)

# In ra các giá trị thực tế từ tập kiểm tra theo mảng với định dạng .2f
actual_prices_formatted = [f"{price:.2f}" for price in df_test['Giá'].values]
# print("\nGiá trị thực tế từ tập kiểm tra:", actual_prices_formatted)

# In ra các chỉ số đánh giá hiệu suất
print(f'\nMean Squared Error trên tập kiểm tra: {mse:.2f}')
print(f'R-squared trên tập kiểm tra: {r2:.2f}')