# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""
from contextlib import contextmanager

@contextmanager
def open_many(files=None, mode="r"):
    """
        使用with 同时打开多个文件并返回数据
    :param files: 文件列表
    :return:
    """
    if files is None:
        files = []
    try:
        fs = []
        for f in files:
            fs.append(open(f, mode))
        yield fs
    except ValueError as e:
        print(e)
    finally:
        for f in fs:
            f.close()
