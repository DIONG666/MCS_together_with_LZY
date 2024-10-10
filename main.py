import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn import metrics

# 导入模型定义和数据读取函数
from Predict.utils.dataset import read_data
from Predict.utils.transformer import Transformer
from Predict.scripts.model.train import opt, predict

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 读取数据
    path = '../../data/BJ16_In.h5'
    X, y, mmn = read_data(path, opt)

    # 划分训练集和测试集
    x_test = X[-opt.test_size:]
    y_test = y[-opt.test_size:]

    test_data = list(zip(*[x_test, y_test]))
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)

    # 定义模型
    input_shape = X.shape
    meta_shape = []
    cross_shape = []

    model = Transformer(input_shape,
                        meta_shape,
                        cross_shape,
                        nb_flows=opt.nb_flow2,
                        fusion=opt.fusion,
                        maps=(opt.meta + opt.cross + 1),
                        d_model=opt.d_model,
                        dk_t=opt.dk_t,
                        dk_s=opt.dk_s,
                        nheads_spatial=opt.nheads_s,
                        nheads_temporal=opt.nheads_t,
                        d_inner=opt.d_inner,
                        layers=opt.layers,
                        flags_meta=opt.meta,
                        flags_cross=opt.cross
                        ).to(device)

    # 定义损失函数
    if opt.loss == 'l1':
        criterion = torch.nn.L1Loss().to(device)
    elif opt.loss == 'l2':
        criterion = torch.nn.MSELoss().to(device)

    pred, truth = predict('test')

    last_ts_pred = pred[-1]
    last_ts_truth = truth[-1]

    print("Prediction for the last time slot:")
    print(last_ts_pred)
    print("Ground truth for the last time slot:")
    print(last_ts_truth)

    mae = metrics.mean_absolute_error(last_ts_truth.ravel(), last_ts_pred.ravel())
    print('MAE for the last time slot: {:.4f}'.format(mae))


