import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_missing_values(stats):
    """
    Vẽ biểu đồ cột thể hiện số lượng giá trị thiếu.
    Args:
        stats (list of dict): Danh sách thống kê giá trị thiếu từ data_processing.
    """
    filtered_stats = [s for s in stats if s['count'] > 0]
    
    if not filtered_stats:
        print("No missing values found.")
        return

    cols = [s['column'] for s in filtered_stats]
    values = [s['count'] for s in filtered_stats]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(cols, values, color='salmon', edgecolor='black', alpha=0.7)
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Number of Missing Values")
    plt.title("Missing Values per Column (Descending Order)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def plot_missing_heatmap(missing_matrix, feature_names):
    """
    Vẽ Heatmap hiển thị phân bố dữ liệu thiếu trên toàn bộ các cột.
    Args:
        missing_matrix (numpy.ndarray): Ma trận boolean (0/1).
        feature_names (list): Tên các cột.
    """
    plt.figure(figsize=(14, 8))
    
    ax = sns.heatmap(missing_matrix, cbar=True, yticklabels=False, cmap='viridis')
    
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['Present (0)', 'Missing (1)'])
    
    plt.xticks(
        ticks=np.arange(len(feature_names)) + 0.5, 
        labels=feature_names, 
        rotation=45, 
        ha='right',
        fontsize=10
    )
    
    plt.title('Missing Values Heatmap', fontsize=14)
    plt.tight_layout()
    plt.show()



def plot_key_distributions(data):
    """
    Vẽ phân phối (Histogram) cho 2 cột số quan trọng nhất:
    - City Development Index
    - Training Hours
    """
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    col_1 = data['city_development_index']
    sns.histplot(col_1, bins=20, kde=True, color='skyblue', edgecolor='black')
    plt.title('Distribution: City Development Index')
    plt.xlabel('City Development Index')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    col_2 = data['training_hours']
    sns.histplot(col_2, bins=30, kde=True, color='salmon', edgecolor='black')
    plt.title('Distribution: Training Hours')
    plt.xlabel('Training Hours')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()



def plot_key_boxplots(data):
    """
    Vẽ 4 biểu đồ Boxplot đặc thù cho HR Analytics (Lưới 2x2):
    - Hàng 1: Tìm Outlier đơn biến.
    - Hàng 2: So sánh phân phối theo Target.
    """
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    sns.boxplot(x=data['city_development_index'], color='skyblue')
    plt.title('Outlier Detection: City Development Index')
    plt.xlabel('City Development Index')

    plt.subplot(2, 2, 2)
    sns.boxplot(x=data['training_hours'], color='salmon')
    plt.title('Outlier Detection: Training Hours')
    plt.xlabel('Training Hours')
    plt.subplot(2, 2, 3)
    sns.boxplot(
        x=data['target'], 
        y=data['city_development_index'], 
        hue=data['target'], 
        legend=False, 
        palette='Set2'
    )
    plt.title('City Index vs Target (0: Stay, 1: Leave)')
    plt.xlabel('Target')
    plt.ylabel('City Development Index')

    plt.subplot(2, 2, 4)
    sns.boxplot(
        x=data['target'], 
        y=data['training_hours'], 
        hue=data['target'], 
        legend=False, 
        palette='Set2'
    )
    plt.title('Training Hours vs Target (0: Stay, 1: Leave)')
    plt.xlabel('Target')
    plt.ylabel('Training Hours')

    plt.tight_layout()
    plt.show()

def plot_key_scatter(data):
    """
    Vẽ Scatterplot duy nhất giữa 2 biến số, tô màu theo Target.
    """
    plt.figure(figsize=(10, 7))
    
    sns.scatterplot(
        x=data['city_development_index'], 
        y=data['training_hours'], 
        hue=data['target'],
        palette='viridis',
        alpha=0.6,
        s=60
    )
    
    plt.title('Correlation: City Index vs Training Hours', fontsize=14)
    plt.xlabel('City Development Index')
    plt.ylabel('Training Hours')
    plt.legend(title='Target (0: Stay, 1: Leave)', loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_categorical_count(data, col_name, title=None):
    """
    Vẽ biểu đồ cột đếm số lượng cho biến phân loại.
    - Tự động chuyển chuỗi rỗng '' thành 'Unknown'.
    - Tự động chọn bảng màu phù hợp.
    """
    col_data = data[col_name]
    unique_vals, counts = np.unique(col_data, return_counts=True)
    labels = []
    for val in unique_vals:
        val_str = str(val).strip()
        if val_str == '' or val_str == 'nan':
            labels.append('Unknown')
        else:
            labels.append(val_str)
            
    sorted_data = sorted(zip(labels, counts), key=lambda x: x[1], reverse=True)
    sorted_labels = [x[0] for x in sorted_data]
    sorted_counts = [x[1] for x in sorted_data]

    if col_name == 'gender':
        colors = ['#4285F4', '#FBBC05', '#34A853', '#EA4335'] 
        if len(sorted_labels) > 4: colors += ['#9E9E9E'] * (len(sorted_labels) - 4)
        colors = colors[:len(sorted_labels)]
    else:
        colors = sns.color_palette('Paired', n_colors=len(sorted_labels))
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_labels, sorted_counts, color=colors, edgecolor='black', alpha=0.85)
    
    if title is None:
        title = f'Distribution of {col_name}'
    
    plt.title(title, fontsize=14)
    plt.xlabel(col_name, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    plt.tight_layout()
    plt.show()

def plot_category_vs_target(stats, col_name, title=None):
    """
    Vẽ biểu đồ dựa trên danh sách thống kê (stats) đã tính toán.
    """
    labels = [s['label'] for s in stats]
    counts_0 = [s['0'] for s in stats]
    counts_1 = [s['1'] for s in stats]
    
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    rects1 = plt.bar(x - width/2, counts_0, width, label='0.0 (Stay)', color='#76b5c5', edgecolor='black')
    rects2 = plt.bar(x + width/2, counts_1, width, label='1.0 (Job Change)', color='#e78ac3', edgecolor='black')

    if title is None:
        title = f'{col_name} vs Target'
    
    plt.title(title, fontsize=14)
    plt.xlabel(col_name, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend(title='Target')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                plt.text(rect.get_x() + rect.get_width()/2., height,
                         f'{int(height)}',
                         ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()
    
    
    
    
    
def plot_experience_boxplot(data, col_name='experience', title='Experience Boxplot'):
    """
    Vẽ boxplot cho biến experience:
    - Tự động chuyển experience từ dạng chuỗi ('<1', '>20', '3', ...) sang dạng số.
    - Bỏ qua các giá trị rỗng hoặc không hợp lệ.
    """
    
    col_data = data[col_name].astype(str)
    col_data = np.char.strip(col_data)
    numeric_vals = np.full(col_data.shape, np.nan, dtype=float)

    mask_lt1 = col_data == '<1'
    numeric_vals[mask_lt1] = 0

    mask_gt20 = col_data == '>20'
    numeric_vals[mask_gt20] = 21


    mask_numeric = np.char.isdigit(col_data)
    numeric_vals[mask_numeric] = col_data[mask_numeric].astype(float)

    numeric_vals_clean = numeric_vals[~np.isnan(numeric_vals)]

    if len(numeric_vals_clean) == 0:
        print("Không có dữ liệu hợp lệ để vẽ boxplot cho experience.")
        return
    plt.figure(figsize=(8, 6))
    plt.boxplot(numeric_vals_clean, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#A1C9F1", color='black'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))

    plt.title(title, fontsize=14)
    plt.ylabel('Years of Experience', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
