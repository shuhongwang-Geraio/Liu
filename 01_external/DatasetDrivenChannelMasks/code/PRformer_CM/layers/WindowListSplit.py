import math

import numpy as np


def splitWindowList(input_list):
    result_lists = []
    gcd_list = []
    input_label = np.zeros(len(input_list))
    input_length = len(input_list)
    # labeledCount = 0 #统计所有被分配的总数 这样不好 有bug
    while input_label.sum() <input_length:
        tmpList = None
        lastValue = None
        for i in range(input_length-1,-1,-1):
            label = input_label[i]
            value = input_list[i]
            if label == 0 or i == 0  : #or ( (lastValue is not None) and (lastValue % value == 0)): # 这个条件目的是多个窗口尽量共享，去掉后 只有第一个窗口可以复用多次
                if tmpList is None:
                    tmpList = [value]
                    input_label[i] = 1
                    lastValue = value
                elif lastValue %value ==0:
                    tmpList.append(value)
                    input_label[i] = 1
                    lastValue = value

        if (tmpList is not None) and len(tmpList)!=0:
            tmpList.sort()
            result_lists.append(tmpList)
    return result_lists

def countResultWindows(result_lists):
    cnt = 0
    for windows in result_lists:
        cnt+=len(windows)
    return cnt

if __name__ == '__main__':
    # 示例用法
    # input_list = [24, 48, 72, 96, 144]
    input_list = [24, 48, 96, 144]
    # input_list = [24, 180, 240, 360, 720] # 通过FFT筛选的 效果不如上边的
    # input_list = [48]
    # input_list = [48,52,96,144]
    result = splitWindowList(input_list) # [[24, 72, 144], [24, 48, 96]]
    print(result)
    print(countResultWindows(result))
    # moniBuild(input_list,result)
#
