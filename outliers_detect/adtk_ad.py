# -*- coding: utf-8 -*-

"""
__author__ == "BigBrother"

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_range = pd.date_range(pd.to_datetime("2020/7/11 10:30:00"), end= pd.to_datetime("2020/7/11 18:00:00"), freq="30T")
x = np.linspace(0, len(data_range), len(data_range))
data = pd.DataFrame({"time": data_range, "value": x})
data.set_index(["time"], inplace=True)
data.iloc[3:4] = 7
data.iloc[10:11] = 3

from adtk.transformer import DoubleRollingAggregate, RollingAggregate
from adtk.detector import ThresholdAD
from adtk.pipe import Pipenet

# 构建异常检测流水线, DoubleRollingAggregate两个滑动窗口并排移动，没有交叉，步长为1，初始的时候左窗口的右边界在数组外，右窗口的左边界在第一个元素
step = {"abs_skipe_change": {"model": DoubleRollingAggregate(agg="mean", window=(1, 1), center=False, diff="l1"),
                            "input": "original"},
        "positive_change": {"model": ThresholdAD(low=0, high=4),
                            "input": "abs_skipe_change"}
}
mypipenet = Pipenet(steps=step)

anomalies = mypipenet.fit_detect(data, return_list=True, return_intermediate=True)
print(anomalies)
from adtk.visualization import plot
plot(data, anomaly=anomalies, anomaly_color='red', ts_markersize=10, anomaly_markersize=15, ts_linewidth=3, anomaly_alpha=1)
plt.show()