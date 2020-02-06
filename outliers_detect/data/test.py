# -*- coding: utf-8 -*-

"""
__author__ == "BigBrother"

"""

import os
import pandas as pd

filenames = os.listdir("./origin/")
for filename in filenames:
    with open("./origin/" + filename, encoding="gbk") as f:
        data = pd.read_csv(f)
        data_ = data[data["label"] == 1]

    data_.to_csv("./cleaned/" + filename, index=False, encoding="gbk")