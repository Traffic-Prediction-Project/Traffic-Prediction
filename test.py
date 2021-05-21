import pandas as pd
import numpy as np


def random_entropy(traffic_info):
    N = np.zeros(traffic_info.shape[1])
    for i in range(traffic_info.shape[1]):
        N[i] = len(np.unique(traffic_info[:, i]))
    s_random = np.log2(N)
    return s_random, N


def shannon_entropy(traffic_info):
    s_shannon = np.zeros(traffic_info.shape[1])
    for i in range(traffic_info.shape[1]):
        unique = np.unique(traffic_info[:, i])
        P = np.zeros(unique.shape[0])
        for j in range(unique.shape[0]):
            P[j] = np.sum(traffic_info[:, i] == unique[j])
        P = P / traffic_info.shape[0]
        s_shannon[i] = -np.sum(P * np.log2(P))
    return s_shannon


def real_entropy(traffic_info):
    s_real = np.zeros(traffic_info.shape[1])
    for i in range(traffic_info.shape[1]):
        unique = np.unique(traffic_info[:, i])
        dictionary = [str(j) for j in unique]
        p = ""
        for j in traffic_info[:, i]:
            pc = p + str(j)
            if pc in dictionary:
                p = pc
            else:
                dictionary.append(pc)
                p = str(j)
        s_t = 0
        for j in dictionary:
            s_t += len(j)
        s_real[i] = traffic_info.shape[0] / s_t * np.log(traffic_info.shape[0])
    return s_real


def get_max_predictability(N, S):
    ans = np.zeros(N.shape)
    for i in range(N.shape):
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
    pd_traffic_info = pd.read_csv('.\\resources\\上海区域交通信息.csv')
    # times = pd_traffic_info.loc[:,"time"]
    # times = pd.DataFrame(times)
    # region = info[0:0]

    # 去除-1占比大于10%的数据
    traffic_info = pd_traffic_info.values[:, 1:]
    delete_list = []
    for i in range(traffic_info.shape[1]):
        mean_value = 0
        mean_sum = 0
        mean = 0
        for j in range(traffic_info.shape[0]):
            if traffic_info[j][i] != -1.0:
                mean_value += traffic_info[j][i]
                mean_sum += 1
        if mean_sum <= 0.9 * traffic_info.shape[0]:
            delete_list.append(i)
        else:
            mean = mean_value / mean_sum
            for j in range(traffic_info.shape[0]):
                if traffic_info[j][i] == -1.0:
                    traffic_info[j][i] = mean
    traffic_info = np.delete(traffic_info, delete_list, axis=1)

    # 离散化
    traffic_info *= 10
    traffic_info = np.trunc(traffic_info)

    # 计算熵
    s_random, N = random_entropy(traffic_info)
    s_shannon = shannon_entropy(traffic_info)
    s_real = real_entropy(traffic_info)

    # 保存中间结果
    pd_s_random = pd.DataFrame(s_random)
    pd_s_shannon = pd.DataFrame(s_shannon)
    pd_s_real = pd.DataFrame(s_real)
    pd_N = pd.DataFrame(N)
    entropy = pd.concat([pd_s_random, pd_s_shannon], axis=1)
    entropy = pd.concat([entropy, pd_s_real], axis=1)
    entropy = pd.concat([entropy, pd_N], axis=1)
    entropy.to_csv(".\\resources\\entropy.csv", index=False)

    # random_pi_max = get_max_predictability(N, s_random)
    # print(random_pi_max)

    # pd_info = pd.DataFrame(traffic_info)
    # pd_info = pd.concat([times, pd_info], axis=1)
    # pd_info = pd.concat([region, pd_info], axis=0)
    # pd_info.to_csv("ans.csv", index=False)
