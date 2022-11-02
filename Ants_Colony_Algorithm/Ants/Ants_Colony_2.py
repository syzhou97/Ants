# -*- coding: utf-8 -*-
import math
import random
import copy
import matplotlib.pyplot as plt

# 地图长度
L = 100
# 地图高度
H = 80
# 城市个数
N = 20
# 蚂蚁个数
M = 20
# 迭代次数
T = 300

# 地图的对角线长度
LH = int(math.sqrt(L * L + H * H))

# 信息素的加权值
alpha = 1
# 能见度的加权值
beta = 2
# 信息素的蒸发率
rho = 0.5

# 城市的横坐标
city_x = [0 for n in range(N)]
# 城市的纵坐标
city_y = [0 for n in range(N)]

# 城市i和城市j之间的距离
distance = [[0 for j in range(N)] for i in range(N)]
# 能见度, 两点之间距离的倒数, 启发信息函数
eta = [[0 for j in range(N)] for i in range(N)]

# 当前时刻, 城市i和城市j之间的道路上信息素的值
tau = [[0 for j in range(N)] for i in range(N)]

# pathlen[t] 第 t 次迭代后得出的路径长度
pathLen = []
# pathCity[t] 第 t 次迭代后得出的路径
pathCity = []

# 第 best[0] 次迭代的路径最短, 全局最短路径的编号
best = [0]
# 全局最短路径上依次经过的城市的横坐标
X = []
# 全局最短路径上依次经过的城市的纵坐标
Y = []


# 初始化
# Step 1: 随机生成所有城市的坐标 (city_x, city_y)
# Step 2: 计算任意两城市之间的距离 (distance) 和能见度 (eta)
# Step 3: 用贪婪算法得出初始路径
# Step 4: 计算得出并记录所有路径的信息素浓度(tau)

def init():
    # ------------------------------------------- Step 1
    # 遍历所有城市
    for n in range(N):
        # 随机横坐标
        x = random.randint(0, L - 1)
        # 随机纵坐标
        y = random.randint(0, H - 1)
        # 记录城市 n 的横坐标
        city_x[n] = x
        # 记录城市 n 的纵坐标
        city_y[n] = y
    # ------------------------------------------- Step 2
    # 从城市 i 出发
    for i in range(N):
        # 到达城市 j
        for j in range(N):
            # 城市 i 和城市 j 之间的距离
            dij = math.pow(city_x[i] - city_x[j], 2)
            dij = dij + math.pow(city_y[i] - city_y[j], 2)
            dij = math.sqrt(dij)
            # 记录两城市之间的距离
            distance[i][j] = dij
            # 计算能见度
            # 如果 i 等于 j
            if i == j:
                # 城市到自己的能见度为 0
                eta[i][j] = 0
            else:
                # 两城市之间的能见度为两城市之间距离的倒数
                eta[i][j] = 1 / dij
    # ------------------------------------------- Step 3
    # 允许去的城市, 即未去过的城市
    # 0: 不允许, 1: 允许
    allow = [1 for n in range(N)]
    # 假设：从城市0出发
    allow[0] = 0
    # 路径
    apath = [0]
    # 当前位置
    apos = 0
    # 下一步去的城市
    acity = 0
    # 与下一个城市的距离
    away = 0
    # 总路径长度
    alen = 0
    #  第 c 次去往下一个城市
    for c in range(N - 1):
        # 设置去往下一个城市的距离, 最大值
        away = LH
        # 选择去往哪一个城市
        for n in range(N):
            # 如果允许去城市 n
            if allow[n] == 1:
                # 如果去城市 n 的距离 小于 当前要去的城市的距离
                if distance[apos][n] < away:
                    # 更新要去的城市
                    acity = n
                    # 更新要去的城市的距离
                    away = distance[apos][n]
        # 更新所在的位置
        apos = acity
        # 更新路径
        apath.append(apos)
        # 更新总路径长度
        alen = alen + away
        # 更新允许去的城市
        allow[apos] = 0
    # 回到出发点
    apath.append(0)
    # 更新总路径长度
    alen = alen + distance[apos][0]
    # 添加初始路径长度
    pathLen.append(alen)
    # 添加初始路径经过的城市
    pathCity.append(copy.deepcopy(apath))
    # ------------------------------------------- Step 4
    # 获得信息素的初始浓度值
    tau0 = M / alen
    # 设置所有城市之间路径的信息素浓度
    for i in range(N):
        for j in range(N):
            tau[i][j] = tau0
    # 城市到本身无路径, 信息素浓度为 0
    for n in range(N):
        tau[n][n] = 0


# 多次迭代
# Step  1: 第 t 次迭代
# Step  2: 第 t 次迭代, 初始化: 所有蚂蚁 (m) 的路径 (path[m][]), 出发城市 (path[m][0]),
#                              允许去的城市 (allowed[m][]) 即本次迭代中未去过的城市
# Step  3: 第 t 次迭代, 第 c 次选择下一个城市
# Step  4: 第 t 次迭代, 第 c 次选择下一个城市, 第 m 只蚂蚁选择下一个城市
# Step  5: 第 t, c, m 中, 初始化：估值函数          pij[0][],
#                               估值的概率占比     pij[1][],
#                               比例选择时的概率点 pij[2][]
# Step  6: 第 t, c, m 中, 获得蚂蚁 m 的允许去的城市 cho = []
# Step  7: 第 t, c, m 中, 计算蚂蚁 m 去城市 n 的 估值函数
# Step  8: 第 t, c, m 中, 计算蚂蚁 m 去城市 n 的 估值函数值 在所有估值函数值之和的占比
# Step  9: 第 t, c, m 中, 计算比例选择 (轮盘赌) 时, 去城市 n 还是城市 n+1 的概率选择点
# Step 10: 第 t, c, m 中, 模仿轮盘赌, 随机比例选择
# Step 11: 第 t 次迭代, 所有蚂蚁回到出发城市, 形成一条首尾相连的路径
# Step 12: 第 t 次迭代, 计算所有蚂蚁 (m) 的路径长度 (mplen[m]), 途经城市 (taumnn[m][][])
# Step 13: 第 t 次迭代, 更新路径 (tau[i][j]) 的信息素 (蒸发剩下的 + 新留下的)
# Step 14: 第 t 次迭代, 选出所有蚂蚁 (m) 路径长度 (mplen[m]) 中的最短路径长度
# Step 15: 第 t 次迭代, 判断是否出现比全局最短路径更短的本次迭代最短路径, 并更新

def iteration():
    # ------------------------------------------- Step 1
    # 第 t 次迭代
    for t in range(1, T + 1):
        # --------------------------------------- Step 2
        # 所有蚂蚁的路径
        path = [[] for m in range(M)]
        # 蚂蚁 m 从 城市 m 出发
        for m in range(M):
            path[m].append(m)
        # 蚂蚁允许去的城市
        allowed = [[0 if i == j else 1 for j in range(N)] for i in range(N)]
        # 第 c 次去往下一个城市, 除了出发城市, 有 N-1 个城市
        # --------------------------------------- Step 3
        for c in range(N - 1):
            # ----------------------------------- Step 4
            # 第 m 只蚂蚁选择下一个城市
            for m in range(M):
                # ------------------------------- Step 5
                # 估值函数, 估值的概率占比, 比例选择时的概率点
                pij = [[0 for j in range(N)] for i in range(3)]
                # 去往的下一个城市的编号
                city = 0
                # 在第 c 次选择城市时, 可选城市的编号
                # ------------------------------- Step 6
                cho = []
                # 判断城市 n 是否可选
                for n in range(N):
                    # 如果可选
                    if allowed[m][n] == 1:
                        # 添加入 cho
                        cho.append(n)
                # ------------------------------- step 7
                # 遍历所有可去城市
                for n in cho:
                    # 蚂蚁 m 所处的当前城市
                    x = path[m][-1]
                    # 蚂蚁 m 下一步去从城市 n 的概率
                    pij[0][n] = math.pow(tau[x][n], alpha)
                    pij[0][n] = pij[0][n] * math.pow(eta[x][n], beta)
                # -------------------------------step 8
                # 求和
                p1 = sum(pij[0])
                # 归一化
                for n in cho:
                    # 蚂蚁 m 去从城市 n 的概率 占 所有概率之和的比例
                    pij[1][n] = pij[0][n] / p1
                # ------------------------------- Step 9
                # 比例选择法（轮盘赌法）的第一个概率点
                p2 = 0
                # 遍历所有可去城市
                for n in cho:
                    # 获得所有概率点
                    pij[2][n] = p2 + pij[1][n]
                    p2 = pij[2][n]
                # ------------------------------- Step 10
                # 模仿轮盘, 随机选择
                rand = random.random()
                # 遍历所有可去城市
                for n in cho:
                    # 如果概率点落在去城市 n 的扇面内
                    if pij[2][n] > rand:
                        # 则去城市 n
                        city = n
                        # 结束遍历
                        break
                # 更新路径
                path[m].append(city)
                # 更新允许去的城市
                allowed[m][city] = 0
        # --------------------------------------- Step 11
        # 回到出发城市
        for m in range(M):
            # 添加路径
            path[m].append(m)
        # --------------------------------------- Step 12
        # 所有蚂蚁走完所有城市的路径长度
        mplen = []
        # 蚂蚁 m 是否经过城市 i 到城市 j 的路径, 留下信息素
        taumnn = []
        # 遍历所有蚂蚁
        for m in range(M):
            # 初始设置: 蚂蚁 m 没有经过城市 i 到城市 j 的路径
            taunn = [[0 for i in range(N)] for j in range(N)]
            # 总路径长度为 0
            plen = 0
            # 遍历蚂蚁 m 经过的城市
            for p in range(N):
                # 出发城市
                x = path[m][p]
                # 到达城市
                y = path[m][p + 1]
                # 更新路径长度
                plen = plen + distance[x][y]
                # 在城市 x 到城市 y 的路径上留下信息素
                taunn[x][y] = 1
            # 更新所有蚂蚁的路径总长度的列表
            mplen.append(plen)
            # 更新所有蚂蚁留下信息素的路径列表
            taumnn.append(copy.deepcopy(taunn))
        # --------------------------------------- Step 13
        # 城市 i 出发
        for i in range(N):
            # 到达城市 j
            for j in range(N):
                # 蚂蚁留下的信息素
                taumij = 0
                # 遍历所有蚂蚁留下信息素的路径
                for m in range(M):
                    # 如果蚂蚁 m 在城市 i 到城市 j 的路径上留下信息素
                    if taumnn[m][i][j] == 1:
                        # 更新该段路径留下的的信息素之和
                        taumij = taumij + 1 / mplen[m]
                # 更新该路径的信息素（蒸发剩下的 + 新留下的）
                tau[i][j] = (1 - rho) * tau[i][j] + taumij
        # --------------------------------------- Step 14
        # 路径总长度的最大值 小于 对角线长度的城市个数倍
        pathlent = LH * N
        # 蚂蚁 ant 在本次迭代中走的路径最短
        ant = 0
        # 遍历所有蚂蚁
        for m in range(M):
            # 如果蚂蚁 m 走的路径长度 小于 本次迭代的最短路径
            if mplen[m] < pathlent:
                # 更新最短路径
                pathlent = mplen[m]
                # 更新蚂蚁编号
                ant = m
        # 添加本次迭代的最短路径
        pathLen.append(pathlent)
        # 添加本次迭代的最短路径经过的城市
        pathCity.append(copy.deepcopy(path[ant]))
        # --------------------------------------- Step 15
        # 判断是否出现比全局最短路径更短的本次迭代最短路径
        if pathlent < pathLen[best[0]]:
            # 更新全局最短路径的编号
            best[0] = len(pathLen)
# 展示结果
# Step 1: 输出 每次迭代的最短路径长度
# Step 2: 输出 全局最短路径长度 及其 首次出现 的 迭代次数
# Step 3: 可视化展示 所有城市 的 位置
# Step 4: 可视化展示 全局最短路径
# Step 5: 可视化展示 每次迭代的最短路径长度 的变化趋势

def showResult():
    # ------------------------------------------- Step 1
    # 遍历所有的迭代结果
    for t in range(T + 1):
        # 输出 t 次迭代的最短路径长度
        print("{0:>3} {1:>16}".format(t, pathLen[t]))
    # ------------------------------------------- Step 2
    # 输出全局最短路径及其长度, 首次出现的迭代次数
    print("\n全局最短路径: ", pathCity[best[0]], " 长度: ", pathLen[best[0]])
    print("首次出现在第 ", best[0], "次迭代")
    # ------------------------------------------- Step 3
    # 画布
    plt.figure(1)
    # 子图 1
    plt.subplot(1,1,1)
    # 散点图展示所有城市的位置
    plt.scatter(city_x, city_y, color='b')
    # 遍历所有城市
    for n in range(N):
        # 城市 n 的坐标信息
        msg = "({},{})".format(city_x[n], city_y[n])
        # 需标注的城市的坐标
        x = city_x[n]
        y = city_y[n]
        # 文本标注的位置
        xt = x + 0.5
        yt = y + 0.5
        # 标注
        plt.annotate(msg, xy=(x, y), xytext=(xt, yt))
    # ------------------------------------------- Step 4
    # 遍历全局最短路径经过的城市
    for p in pathCity[best[0]]:
        # 横坐标
        X.append(city_x[p])
        # 纵坐标
        Y.append(city_y[p])
    # 折线图展示: 全局最短路径
    plt.plot(X, Y, color='c')
    # x 轴标签
    plt.xlabel("city_x")
    # y 轴标签
    plt.ylabel("city_y")
    # 标题
    plt.title("Global shortest path")
    # 添加网格线
    plt.grid()
    # ------------------------------------------- Step 5
    # 子图 2
    plt.figure(2)
    plt.subplot(1,1,1)
    # 折线图展示: 每次迭代路径最短长度的变化趋势
    plt.plot([t for t in range(T + 1)], pathLen)
    # x 轴标签
    plt.xlabel("T iteration")
    # y 轴标签
    plt.ylabel("The shortest length")
    # 标题
    plt.title("Changing trend")
    # 展示输出
    plt.show()

# 初始化
init()
# 多次迭代
iteration()
# 展示结果
showResult()


