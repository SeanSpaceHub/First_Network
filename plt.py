
from matplotlib import pyplot as plt
import numpy as np
import math


def plt_distribution(for_show, name='items', y_range=None, *, sub=False, x_limit=None, y_limit=None):
    if x_limit:
        min = x_limit[0]
        max = x_limit[1]
    else:
        min = for_show.min()
        max = for_show.max()
    size = for_show.size
    step = (max - min) / 100
    x = np.arange(min, max, step)
    y = np.zeros_like(x)
    for i in range(100):
        y[i] = np.sum(((min + i * step) <= for_show) & (for_show < (min + (i + 1) * step))) / size
    if x_limit:
        y[0] += np.sum(for_show < min) / size
        y[99] += np.sum(for_show >= max) / size
    plt.title("distribution of %s" % name)
    plt.xlabel("value")
    plt.ylabel("probability")
    plt.plot(x, y, label=name)
    if x_limit:
        plt.xlim(*x_limit)
    if y_limit:
        plt.ylim(*y_limit)
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])
    plt.legend()
    if not sub:
        plt.show()


def plt_change_process(for_show_turple, Lable_turple, show_para_turple=None, *, sub=False, x_limit=None, y_limit=None):
    num = len(for_show_turple)
    if isinstance(for_show_turple[0], (tuple, list)):
        x = np.arange(1, len(for_show_turple[0])+1, 1, dtype=int)
    else :
        x = np.arange(1, for_show_turple[0].size + 1, 1, dtype=int)
    name = ''
    for i in Lable_turple:
        name = name + ' ' + i
    plt.title("change_process of %s" % name)
    plt.xlabel("times")
    plt.ylabel("value")
    if x_limit:
        plt.xlim(*x_limit)
    if y_limit:
        plt.ylim(*y_limit)
    for i in range(num):
        if show_para_turple is not None:
            plt.plot(x, for_show_turple[i], show_para_turple[i], label=Lable_turple[i])
        else:
            plt.plot(x, for_show_turple[i], label=Lable_turple[i])
    plt.legend()
    if not sub:
        plt.show()


def plt_sub(f_turple, parameter_turple, *,  x_limit=None, y_limit=None):
    num = len(parameter_turple)
    broadcast = (len(f_turple) != num)
    length = int(math.sqrt(num)+0.499)
    width = int(num / length + 0.99)
    for i in range(0, num):
        plt.subplot(length, width, i+1)
        if broadcast:
            f_turple[0](*parameter_turple[i], sub=True, x_limit=x_limit, y_limit=y_limit)
        else:
            f_turple[i](*parameter_turple[i], sub=True, x_limit=x_limit, y_limit=y_limit)
    plt.legend()
    plt.tight_layout()
    plt.show()


'''
'-'	实线样式
'--'	短横线样式
'-.'	点划线样式
':'	虚线样式
'.'	点标记
','	像素标记
'o'	圆标记
'v'	倒三角标记
'^'	正三角标记
'&lt;'	左三角标记
'&gt;'	右三角标记
'1'	下箭头标记
'2'	上箭头标记
'3'	左箭头标记
'4'	右箭头标记
's'	正方形标记
'p'	五边形标记
'*'	星形标记
'h'	六边形标记 1
'H'	六边形标记 2
'+'	加号标记
'x'	X 标记
'D'	菱形标记
'd'	窄菱形标记
'&#124;'	竖直线标记
'_'	水平线标记

'b'	蓝色
'g'	绿色
'r'	红色
'c'	青色
'm'	品红色
'y'	黄色
'k'	黑色
'w'	白色
'''
