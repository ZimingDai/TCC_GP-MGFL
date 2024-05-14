import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Arial'


def draw_data(file_count, str_prefix, address, label, start_row, end_row, color, linestyle):
    df_list = []
    for i in range(1, file_count + 1):
        file_path = f"{address}/{str_prefix}{i}.txt"
        df = pd.read_csv(file_path, header=None)
        
        # 验证 start_row 和 end_row 的有效性
        if end_row < start_row:
            raise ValueError("结束行数必须大于开始行数")
        
        # 选择指定范围的行，并处理数据
        df = df.iloc[start_row:end_row, 0].str.strip().str[10:].astype(float)
        df.rename('col', inplace=True)
        
        df_list.append(df)

    # 将所有文件的数据合并并计算均值
    alpha = pd.concat(df_list, axis=1).mean(axis=1)

    x = range(len(alpha))

    plt.plot(x, alpha, color=color, label=label, linestyle=linestyle, alpha=0.7)
    plt.legend()
    plt.show()


def check_txt_row_consistency(path):
    row_counts = []
    txt_files = [file for file in os.listdir(path) if file.endswith('.txt')]

    if not txt_files:
        raise FileNotFoundError("No .txt files found in the specified directory.")

    for file in txt_files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path, header=None)
        row_counts.append(len(df))

    min_row_count = min(row_counts)
    all_equal = all(count == row_counts[0] for count in row_counts)

    if all_equal:
        print(f"All files have consistent row counts: {min_row_count}")
    else:
        print(f"Row counts are not consistent. Minimum row count is: {min_row_count}")
        
    return min_row_count






if __name__ == '__main__':
    FILE_ADDRESS='../log'
    SAVE_ADDRESS='../analysis'
    plt.figure(figsize=(8, 6))
    num = check_txt_row_consistency(FILE_ADDRESS)
    nodes = 30
    epoch = 10
    start_row = 15
    end_row = 50
    save_name = "mnist"
    
    
    draw_data(file_count=nodes // 2, str_prefix='small_coarse', address=FILE_ADDRESS, label=f'{epoch} epoch, {nodes} nodes',start_row=start_row, end_row=end_row, color=(0 / 255, 0 / 255, 139 / 255), linestyle='-')
    
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Accuracy', fontsize=24)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # 显示图例
    legend = plt.legend(loc='lower right')
    for label in legend.get_texts():
        label.set_fontsize(20)  # 设置图例字体大小
        # label.set_weight('bold')  # 设置图例字体加粗

    plt.savefig(f'{SAVE_ADDRESS}/{save_name}_{nodes}_{epoch}_1.png', dpi=300, bbox_inches='tight')
