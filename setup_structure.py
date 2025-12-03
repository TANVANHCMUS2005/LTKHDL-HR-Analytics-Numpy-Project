import os

# Danh sách thư mục cần tạo
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src"
]

# Danh sách file cần tạo
files = [
    "requirements.txt",
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_preprocessing.ipynb",
    "notebooks/03_modeling.ipynb",
    "src/__init__.py",
    "src/data_processing.py",
    "src/visualization.py",
    "src/models.py"
]

# Tạo thư mục
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Đã tạo folder: {folder}")

# Tạo file rỗng
for file in files:
    with open(file, 'w') as f:
        pass # Chỉ tạo file rỗng
    print(f"Đã tạo file: {file}")