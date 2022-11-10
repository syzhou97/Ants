import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt

# 参数
Np = 50  # 蚂蚁数
G = 300  # 迭代数
alpha = 1
beta = 3
Q = 1
rho = 0.1  # 信息素蒸发系数
# 蚁量模型
info_q = {}  # 每次迭代最优路径长度
path_q = {}  # 每次迭代最优路径序列

best = 1e5  # 全局最优路径长度
cities_order = []  # 全局最优路径序列
ants = []  # 蚁群
pher = []  # 信息素矩阵
cities_dis = []  # 城市距离矩阵
cities_dis_recip = []  # 城市距离倒数矩阵

# 位置信息
locations = [
    [18, 54], [87, 76], [74, 78], [71, 71], [25, 38],
    [58, 35], [4, 50], [13, 40], [18, 40], [24, 42],
    [71, 44], [64, 60], [68, 58], [83, 69], [58, 69],
    [54, 62], [51, 67], [37, 84], [41, 94], [2, 99],
    [7, 64], [22, 60], [25, 62], [62, 32], [87, 7],
    [91, 38], [83, 46], [41, 26], [45, 21], [44, 35]
]


# 初始化
def initialization():
    global pher
    global ants
    global cities_dis
    global cities_dis_recip
    # 初始化城市间路径信息素矩阵
    pher = np.array([1 for i in range(30)] * 30).reshape(30, 30)
    # 初始化蚂蚁群
    for i in range(Np):
        start = random.randint(0, 29)
        tmp = {'tabu': [start], 'Jk': [i for i in range(30) if i != start], 'L': 0}
        ants.append(tmp)
    # 初始化城市间距离矩阵
    cities_dis = np.array([0.1 for i in range(30)] * 30).reshape(30, 30)
    for i in range(30):
        for j in range(i + 1, 30):
            tmp = dist(locations[i], locations[j])
            cities_dis[i][j] = tmp
            cities_dis[j][i] = tmp
    # 城市间距离倒数矩阵
    cities_dis_recip = 1 / cities_dis


# 快速幂运算
def pow_mat(mat, p):
    tmp = deepcopy(mat)
    if p == 1:
        return tmp
    j = 1
    while j < p:
        j = j << 1
        tmp *= tmp
        if p - j == 1:
            tmp *= mat
            break
    return tmp


# 距离计算
def dist(x, y):
    return round(((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5, 2)


# 轮盘赌选择实现
def choice(rs):
    p = random.random()
    i = 0
    while p > 0:
        p -= rs[i]
        i += 1
    return i - 1


# 蚂蚁决策
def decision(city_now, jk, alp, beta):
    global pher
    global cities_dis_recip
    tmp_recip = []
    tmp_pheno = []
    for item in jk:
        tmp_recip.append(cities_dis_recip[city_now][item])
        tmp_pheno.append(pher[city_now][item])
    tmp_recip = np.array(tmp_recip)
    tmp_pheno = np.array(tmp_pheno)
    tmp_recip = pow_mat(tmp_recip, beta)
    tmp_pheno = pow_mat(tmp_pheno, alp)
    p = tmp_recip * tmp_pheno
    s = np.sum(p)
    p /= s
    id = choice(p)
    return jk[id]


# 蚂蚁周游
def walk_cycle(ant, alpha, beta):
    global cities_dis
    while ant['Jk'] != []:
        id = decision(ant['tabu'][-1], ant['Jk'], alpha, beta)
        ant['Jk'].remove(id)
        ant['L'] += cities_dis[ant['tabu'][-1], id]
        ant['tabu'].append(id)
    ant['L'] += cities_dis[ant['tabu'][-1], 0]


# 重置函数
def reset_ant(ant):
    start = random.randint(0, 29)
    ant['tabu'] = [start]
    ant['Jk'] = [i for i in range(30) if i != start]
    ant['L'] = 0


def iteration():
    global best
    global pher
    global info_q, path_q, cities_order
    # global G
    iters = G
    while iters > 0:
        iters -= 1
        tmp = 1e5  # 第G次迭代最优路径长度
        # 蚂蚁周游 并记录最优路径
        for i in range(Np):
            walk_cycle(ants[i], alpha, beta)
            if ants[i]['L'] < best:
                best = ants[i]['L']
                cities_order = ants[i]['tabu']
            if ants[i]['L'] < tmp:
                tmp = ants[i]['L']
                path_q[G - iters] = deepcopy(ants[i]['tabu'])
        info_q[G - iters] = tmp
        # 信息素挥发
        pher = (1 - rho) * pher
        # 按照蚁周模型更新信息素
        for i in range(Np):
            for j in range(30):
                m, n = ants[i]['tabu'][j - 1], ants[i]['tabu'][j]
                # 蚁量模型
                # pher[m][n] += Q/cities_dis[m][n]
                # pher[n][m] += Q/cities_dis[m][n]
                # 蚁密模型
                '''pher[m][n] += Q
                pher[n][m] += Q'''
                # 蚁周模型
                pher[m][n] += Q / ants[i]['L']
                pher[n][m] += Q / ants[i]['L']
        # 清空蚂蚁禁忌表并随机开始城市
        for i in range(Np):
            reset_ant(ants[i])


def showresult():
    iteration_x = list(info_q.keys())
    iteration_y = list(info_q.values())
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.figure(1)
    plt.subplot(1, 1, 1)
    plt.xlabel('Iterations')
    plt.ylabel('Distance of shortest path')
    plt.title('iterations={} Np={} alpha={} beta={} Q={} rho={} min_dist={}'.format(G, Np, alpha, beta, Q, rho, best))
    plt.plot(iteration_x, iteration_y)

    city_x = []
    city_y = []
    for i in range(len(cities_order)):
        city_x.append(locations[cities_order[i]][0])
        city_y.append(locations[cities_order[i]][1])
    plt.figure(2)
    plt.subplot(1, 1, 1)
    # 散点图展示所有城市的位置
    plt.scatter(city_x, city_y, color='b')
    # 城市坐标
    for i in range(len(cities_order)):
        # 城市 n 的坐标信息
        msg = "({},{})".format(city_x[i], city_y[i])
        # 需标注的城市的坐标
        x = city_x[i]
        y = city_y[i]
        # 文本标注的位置
        xt = x + 0.5
        yt = y + 0.5
        # 标注
        plt.annotate(msg, xy=(x, y), xytext=(xt, yt))
    # 折线图展示: 全局最短路径
    city_x.append(city_x[0])
    city_y.append(city_y[0])
    plt.plot(city_x, city_y, color='c')
    # x 轴标签
    plt.xlabel("city_x")
    # y 轴标签
    plt.ylabel("city_y")
    # 标题
    plt.title('Global shortest path  min_distance={}'.format(best))
    # 添加网格线
    plt.grid()
    plt.show()


# 可视化多组数据对比
def showresult_1(it_x, it_y):
    # iteration_x = list(info_q.keys())
    # iteration_y = list(info_q.values())
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.figure(1)
    plt.subplot(1, 1, 1)
    plt.xlabel('Iterations')
    plt.ylabel('Distance of shortest path')
    plt.title('iterations={}  beta={} Q={} rho={} min_dist={}'.format(G, beta, Q, rho, best))
    plt.plot(it_x[0], it_y[0], 'b', label='alpha=0.1')
    plt.plot(it_x[1], it_y[1], '#FFA500', label='alpha=1')
    plt.plot(it_x[2], it_y[2], 'g', label='alpha=10')
    # plt.plot(it_x[3], it_y[3], '#FFA500', label='Np=200')
    plt.legend()
    plt.show()
    # beta = {}  beta,

# 数据对比函数
def muli_image():
    temp_list = [0.1, 1, 10]
    it_x = []
    it_y = []
    global alpha
    for i in range(len(temp_list)):
        alpha = temp_list[i]
        initialization()
        iteration()
        it_x.append(list(info_q.keys()))
        it_y.append(list(info_q.values()))
    showresult_1(it_x, it_y)


if __name__ == '__main__':
    initialization()
    iteration()
    showresult()
# 数据对比
# muli_image()
