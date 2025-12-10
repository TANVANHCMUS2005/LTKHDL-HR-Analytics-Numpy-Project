import numpy as np
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    try:
        data = np.genfromtxt(
            file_path, 
            delimiter=',', 
            dtype=None,       
            names=True,       
            encoding='utf-8'
        )
        print(f"Data loaded successfully. ")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def count_duplicates(data):
    unique_rows = np.unique(data)
    return len(data) - len(unique_rows)

def get_unique_stats(data):
    stats = []
    for col_name in data.dtype.names:
        column_data = data[col_name]
        unique_vals = np.unique(column_data)
        
        if column_data.dtype.kind in ['U', 'S']:
            mask_clean = (np.char.strip(unique_vals) != '') & (np.char.lower(unique_vals) != 'nan')
            clean_unique_vals = unique_vals[mask_clean]
        else:
            clean_unique_vals = unique_vals
            
        stats.append({
            'column': col_name,
            'count': len(clean_unique_vals),
            'examples': clean_unique_vals[:3]
        })
    return stats

def get_missing_stats(data):
    total_rows = data.shape[0]
    stats = []
    
    for col_name in data.dtype.names:
        col = data[col_name]
        
        if col.dtype.kind in ['U', 'S']:
            missing_num = np.sum(np.char.strip(col) == '')
        else:
            try:
                missing_num = np.sum(np.isnan(col))
            except:
                missing_num = 0
                
        percent = (missing_num / total_rows) * 100
        stats.append({
            'column': col_name,
            'count': missing_num,
            'percent': percent
        })
        
    stats.sort(key=lambda x: x['count'], reverse=True)
    
    return stats

def get_missing_matrix(data):
    """
    Tạo ma trận boolean thể hiện vị trí dữ liệu thiếu.
    1 (True) = Missing, 0 (False) = Present.
    """
    num_rows = data.shape[0]
    num_cols = len(data.dtype.names)
    
    # Tạo ma trận chứa toàn số 0 (Kích thước: số dòng x số cột)
    missing_matrix = np.zeros((num_rows, num_cols), dtype=int)
    feature_names = list(data.dtype.names)
    
    for i, col_name in enumerate(feature_names):
        col = data[col_name]
        
        # Tạo mask cho từng cột
        if col.dtype.kind in ['U', 'S']:
            is_missing = (np.char.strip(col) == '')
        else:
            try:
                is_missing = np.isnan(col)
            except:
                is_missing = np.zeros(num_rows, dtype=bool)
        
        # Gán vào cột tương ứng trong ma trận (True thành 1, False thành 0)
        missing_matrix[:, i] = is_missing.astype(int)
        
    return missing_matrix, feature_names

def get_category_stats(data, col_name, target_col='target'):
    """
    Tính toán thống kê Target (0/1) theo từng nhóm phân loại.
    """
    col_data = data[col_name]
    target_data = data[target_col]
    unique_vals = np.unique(col_data)
    
    stats = []
    for val in unique_vals:
        # Xử lý nhãn rỗng
        val_str = str(val).strip()
        label = 'Unknown' if (val_str == '' or val_str == 'nan') else val_str
        
        # Đếm số lượng
        mask = (col_data == val)
        c0 = np.sum((target_data[mask] == 0))
        c1 = np.sum((target_data[mask] == 1))
        total = c0 + c1
        pct = (c1 / total * 100) if total > 0 else 0
        
        stats.append({
            'label': label,
            '0': c0,
            '1': c1,
            'total': total,
            'pct': pct
        })
    
    # Sắp xếp giảm dần theo tổng số lượng
    stats.sort(key=lambda x: x['total'], reverse=True)
    return stats



#---------------------------------------------------------------------------------------------------------------------
# --- 1. CLEANING & CONVERSION FUNCTIONS ---

def convert_experience(col):
    """
    Chuyển đổi cột experience (string) sang số (float).
    Logic: '<1' -> 0, '>20' -> 21, số giữ nguyên, còn lại -> NaN
    """
    # Chuyển về string chuẩn, loại bỏ khoảng trắng
    col_str = np.char.strip(col.astype(str))
    
    # Tạo mảng kết quả mặc định là NaN
    mapped = np.full(col_str.shape, np.nan, dtype=float)
    
    # Áp dụng quy tắc
    mapped[col_str == '<1'] = 0
    mapped[col_str == '>20'] = 21
    
    # Xử lý các giá trị là số
    numeric_mask = np.char.isdigit(col_str)
    mapped[numeric_mask] = col_str[numeric_mask].astype(float)
    
    return mapped

def convert_last_new_job(col):
    """
    Chuyển đổi cột last_new_job sang số.
    Logic: 'never' -> 0, '>4' -> 5, số giữ nguyên.
    """
    col_str = np.char.strip(col.astype(str))
    mapped = np.full(col_str.shape, np.nan, dtype=float)
    
    mapped[col_str == 'never'] = 0
    mapped[col_str == '>4'] = 5
    
    numeric_mask = np.char.isdigit(col_str)
    mapped[numeric_mask] = col_str[numeric_mask].astype(float)
    
    return mapped

# --- 2. ENCODING FUNCTIONS (NUMPY ONLY) ---

def ordinal_encode(col, mapping):
    """
    Mã hóa biến có thứ tự (Ordinal Encoding).
    Args:
        col: Mảng numpy chứa chuỗi
        mapping: Dictionary map giá trị {'Graduate': 0, 'Masters': 1...}
    """
    col_str = np.char.strip(col.astype(str))
    # Mặc định gán -1 cho các giá trị không có trong mapping
    encoded = np.full(col_str.shape, -1, dtype=int)
    
    for key, val in mapping.items():
        encoded[col_str == key] = val
        
    return encoded

def one_hot_encode(col):
    """
    Mã hóa One-Hot cho biến định danh (Nominal).
    Returns: Ma trận 0/1 và danh sách tên các cột mới.
    """
    unique_vals = np.unique(col)
    # Loại bỏ giá trị 'nan' nếu không muốn tạo cột riêng cho nó
    unique_vals = unique_vals[unique_vals != 'nan'] 
    
    # Broadcasting để tạo ma trận One-Hot: (N, 1) == (1, K) -> (N, K) boolean -> int
    one_hot_matrix = (col[:, None] == unique_vals[None, :]).astype(int)
    
    return one_hot_matrix, unique_vals

# --- 3. SCALING FUNCTIONS ---

def standard_scale(X):
    """
    Chuẩn hóa Z-score (Standardization): (X - mean) / std
    Giúp đưa các biến về cùng một phân phối chuẩn (mean=0, std=1).
    """
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    
    # Tránh chia cho 0 nếu std = 0 (cột constant)
    std[std == 0] = 1
    
    return (X - mean) / std