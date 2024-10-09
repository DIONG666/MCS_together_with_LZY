import h5py
import pandas as pd

# 打开h5文件
file_path = '../RawData/BJ16In.h5'  # 替换为你的h5文件路径
output_folder_path = '../DataCSV/'
with h5py.File(file_path, 'r') as f:

    # 创建空的DataFrame存储数据
    data_tables = []

    for i in range(len(f['data'])):
        # 提取数据
        data = f['data'][i]
        # 将数据转换为DataFrame
        df = pd.DataFrame(data)
        df.to_csv(output_folder_path + f'time_slot_{i}.csv', index=False)

        print(f"时隙{i}数据已成功提取并存储为CSV文件。")
