import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def thefunc(x, n, s):
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x) + (
        1 - x) * np.log2(n - 1) - s


def get_max_predictability(N, S):
    ans = np.zeros(N.shape)
    for i in range(len(N)):
        x1 = 0.0
        x2 = 1.0
        s = 0.0
        while x2 - x1 > 1e-4:
            x0 = (x1 + x2) / 2
            if -x0 * np.log2(x0) - (1 - x0) * np.log2(1 - x0) + (
                    1 - x0) * np.log2(N[i] - 1) - S[i] > 0:
                x1 = x0
            else:
                x2 = x0
            s = x0
        ans[i] = s
    return ans


if __name__ == '__main__':
    x = [i for i in range(0, 100, 1)]
    x = np.array(x)
    x = x / 100
    y = thefunc(x, 7, 2.807354922)
    print(y.shape)
    plt.plot(x, y)
    plt.show()

    # intermediate_result = pd.read_csv('.\\resources\\ans.csv')
    # s_random = intermediate_result.values[:, 0:1].reshape(-1)
    # s_shannon = intermediate_result.values[:, 1:2].reshape(-1)
    # s_real = intermediate_result.values[:, 2:3].reshape(-1)
    # N = intermediate_result.values[:, 3:4]
    # N = intermediate_result.values[:, 3:4].reshape(-1)
    # ans = thefunc(0.79, 9, 1.40)
    # # ans = thefunc(1e-7, N[0], s_random[0])
    # print(ans)
    # # ans = get_max_predictability(N, s_random)
    # # print("---")
    # # print(ans)

    # filename = ".\\resources\\区域位置信息.txt" # txt文件和当前脚本在同一目录下，所以不用写具体路径
    # position = []
    # with open(filename, 'r') as file_to_read:
    #     lines = file_to_read.read()
    #     lines = lines.replace('[','')
    #     lines = lines.replace(']','')
    #     lines = lines.replace('\'','')
    #     lines = lines.replace(';', ',')
    #     for i in lines.split(', '):
    #         temp = [float(j) for j in i.split(',')]
    #         mean_temp = [(temp[0]+temp[2])/2, (temp[1]+temp[3])/2]
    #         position.append(mean_temp)

    # position = np.array(position)
    # position = np.concatenate((position, N), axis=1)
    # print(position.shape)

    # data = (
    #     np.random.normal(size=(100, 3)) *
    #     np.array([[0.1, 0.1, 0.1]]) +
    #     np.array([[40, 116.5, 1]])
    # )
    # print(data.shape)
    # data = data.tolist()
    # print(data)
