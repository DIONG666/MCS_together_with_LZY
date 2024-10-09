import h5py
import numpy as np

with h5py.File('BJ16_In.h5', "r") as f:
    data = f['data'][:]  # 读取整个数据集，转换为numpy数组
    print("读取的数据形状:", data.shape)  # 检查数据形状

