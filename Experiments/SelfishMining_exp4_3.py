import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random


class SelfishMining:
    def __init__(self, selfish_OnTimes, selfish_OnProportions, selfish_Total, honest_OnTimes, honest_OnProportions,
                 honest_Total, weight, steps, expCount):
        '''
        构造函数
        :param selfish_OnTimes: 自私玩家的矿机开机时间序列
        :param selfish_OnProportions: 自私玩家的矿机开机比例序列
        :param selfish_Total: 自私玩家的矿机总数
        :param honest_OnTimes: 诚实玩家的矿机开机时间序列
        :param honest_OnProportions: 诚实玩家的矿机开机比例序列
        :param honest_Total: 诚实玩家的矿机总数
        :param weight: 权值
        :param STEPS: 单次模拟实验的步数
        :param expCount: 实验模拟总次数
        '''''
        # 自私玩家的矿机开机时间序列
        self.selfish_OnTimes = selfish_OnTimes
        # 自私玩家的矿机开机比例序列
        self.selfish_OnProportions = selfish_OnProportions
        # 自私玩家的矿机总数
        self.selfish_Total = selfish_Total
        # 诚实玩家的矿机开机时间序列
        self.honest_OnTimes = honest_OnTimes
        # 诚实玩家的矿机开机比例序列
        self.honest_OnProportions = honest_OnProportions
        # 诚实玩家的矿机总数
        self.honest_Total = honest_Total
        # 绘图时 x 轴的最大长度
        self.maxX = 5
        # 权值
        self.gamma = weight
        # 单次模拟实验的步数
        self.steps = steps
        # 实验模拟总次数
        self.markovExpCount = expCount
        # 马尔可夫转移过程的状态字典
        self.stateDict = {}

        '''
        以下变量在执行程序之前均为空值
         '''

        # 系统角度的矿机开机时间序列
        self.god_OnTimes = []
        # 系统角度的矿机开机比例序列
        self.god_OnProportions = []
        # 系统角度下每次开机时自私玩家对应的开机占比
        self.selfish_MachinePortions = []
        # 系统角度下每次开机时诚实玩家对应的开机占比
        self.honest_MachinePortions = []
        # 概率密度函数的 x 坐标（二维锯齿阵）
        self.xList_PDF = []
        # 概率密度函数的 y 坐标（二维锯齿阵）
        self.yList_PDF = []
        # 概率分布函数的 x 坐标（二维锯齿阵）
        self.xList_CDF = []
        # 概率分布函数的 y 坐标（二维锯齿阵）
        self.yList_CDF = []
        # 概率分布函数在各开机节点处的值
        self.CDF_Stage = []
        # 自私玩家在各阶段的概率
        self.selfish_Probability = []
        # 诚实玩家在各阶段的概率
        self.honest_Probability = []
        # 存放不同时间的频数容器
        self.probabilityBox = []
        # 玩家转移概率（默认参数为自私玩家的转移概率）
        self.alpha = 0

    def __getGodView(self, printFlag = False):
        '''
        将自私和诚实玩家的开机策略转换为系统角度下的开机情况
        :return:
        '''
        # 数据格式检测
        selfishLen = len(self.selfish_OnTimes)
        honestLen = len(self.honest_OnTimes)
        if selfishLen != len(self.selfish_OnProportions) or honestLen != len(self.honest_OnProportions):
            print(
                "\n************************************************************************************************\n")
            print(
                "The startup time sequence of machine and its corresponding quantity sequence have different lengths!")
            print(
                "\n************************************************************************************************\n")
            return
        # 自私玩家的总矿机占比
        selfish_Portion = self.selfish_Total / (self.selfish_Total + self.honest_Total)
        # 诚实玩家的总矿机占比
        honest_Portion = self.honest_Total / (self.selfish_Total + self.honest_Total)
        # 根据两个玩家的时间序列进行融合
        pos1 = pos2 = 0
        while pos1 < selfishLen and pos2 < honestLen:
            if self.selfish_OnTimes[pos1] < self.honest_OnTimes[pos2]:
                self.god_OnTimes.append(self.selfish_OnTimes[pos1])
                self.god_OnProportions.append(self.selfish_OnProportions[pos1] * selfish_Portion)
                self.selfish_MachinePortions.append(
                    sum(self.selfish_OnProportions[:pos1 + 1]) * selfish_Portion / sum(self.god_OnProportions))
                self.honest_MachinePortions.append(
                    sum(self.honest_OnProportions[:pos2]) * honest_Portion / sum(self.god_OnProportions))
                pos1 += 1
            elif self.selfish_OnTimes[pos1] > self.honest_OnTimes[pos2]:
                self.god_OnTimes.append(self.honest_OnTimes[pos2])
                self.god_OnProportions.append(self.honest_OnProportions[pos2] * honest_Portion)
                self.selfish_MachinePortions.append(
                    sum(self.selfish_OnProportions[:pos1]) * selfish_Portion / sum(self.god_OnProportions))
                self.honest_MachinePortions.append(
                    sum(self.honest_OnProportions[:pos2 + 1]) * honest_Portion / sum(self.god_OnProportions))
                pos2 += 1
            else:
                self.god_OnTimes.append(self.honest_OnTimes[pos2])
                self.god_OnProportions.append(
                    self.selfish_OnProportions[pos1] * selfish_Portion + self.honest_OnProportions[
                        pos2] * honest_Portion)
                self.selfish_MachinePortions.append(
                    sum(self.selfish_OnProportions[:pos1 + 1]) * selfish_Portion / sum(self.god_OnProportions))
                self.honest_MachinePortions.append(
                    sum(self.honest_OnProportions[:pos2 + 1]) * honest_Portion / sum(self.god_OnProportions))
                pos1 += 1
                pos2 += 1
            # 执行一次后将“初次标记”置为假
            FirstFlag = False
        while pos1 < selfishLen:
            self.god_OnTimes.append(self.selfish_OnTimes[pos1])
            self.god_OnProportions.append(self.selfish_OnProportions[pos1] * selfish_Portion)
            self.selfish_MachinePortions.append(
                sum(self.selfish_OnProportions[:pos1 + 1]) * selfish_Portion / sum(self.god_OnProportions))
            self.honest_MachinePortions.append(
                sum(self.honest_OnProportions[:pos2]) * honest_Portion / sum(self.god_OnProportions))
            pos1 += 1
        while pos2 < honestLen:
            self.god_OnTimes.append(self.honest_OnTimes[pos2])
            self.god_OnProportions.append(self.honest_OnProportions[pos2] * honest_Portion)
            self.selfish_MachinePortions.append(
                sum(self.selfish_OnProportions[:pos1]) * selfish_Portion / sum(self.god_OnProportions))
            self.honest_MachinePortions.append(
                sum(self.honest_OnProportions[:pos2 + 1]) * honest_Portion / sum(self.god_OnProportions))
            pos2 += 1
        if printFlag:
            print("\n汇总得到的时间和开机序列情况如下：")
            print("开机时间序列：%s"%str(self.god_OnTimes))
            print("开机比例序列：%s"%str(self.god_OnProportions))

    def __exppdf(self, x, mu):
        '''
        指数分布的概率密度函数
        :return:
        '''
        return (1 / mu) * np.exp(- (x / mu))

    def __getPDF(self, printFlag = False):
        '''
        计算概率密度函数
        :return:
        '''
        # 设置概率密度的最长 x 坐标以及细化的 x 轴步长
        maxX, steps = 10, 0.001
        # 计算原始概率密度函数
        # 如果开机时间长度为 1
        if len(self.god_OnTimes) == 1:
            # 若首次开机时间为 0 0（则概率密度为标准指数分布 lambda = 1）
            if self.god_OnTimes[0] <= 0.0001:
                self.xList_PDF.append(np.linspace(0, maxX, 10001))
                self.yList_PDF.append(np.exp(-self.xList_PDF[0]))
            # 首次开机时间不为 0，即 [0, onTime, 1] 的情况
            else:
                # 第一个区间段的概率密度为 0
                pos = 0
                x, y = [], []
                while pos < self.god_OnTimes[0]:
                    x.append(pos)
                    y.append(0.0)
                    pos += steps
                self.xList_PDF.append(np.array(x))
                self.yList_PDF.append(np.array(y))
                # 接下来区间的概率密度为偏置的指数分布
                x, y = [], []
                mu = 1 - self.god_OnTimes[0]
                while pos <= maxX:
                    x.append(pos)
                    y.append(self.__exppdf(pos - self.god_OnTimes[0], mu))
                    pos += steps
                self.xList_PDF.append(np.array(x))
                self.yList_PDF.append(np.array(y))
        # 如果开机时间长度大于 1
        if len(self.god_OnTimes) > 1:
            #  若首次开机时间为 0
            if self.god_OnTimes[0] <= 0.0001:
                # 第一阶段概率密度为标准指数分布
                pos = 0
                x, y = [], []
                while pos < self.god_OnTimes[1]:
                    x.append(pos)
                    y.append(np.exp(-pos))
                    pos += steps
                self.xList_PDF.append(np.array(x))
                self.yList_PDF.append(np.array(y))
                # 接下来区间的概率密度为
                mu = 1
                for porIndex in range(1, len(self.god_OnProportions)):
                    # 两个阶段的机器比例
                    coef = sum(self.god_OnProportions[:porIndex+1]) / sum(self.god_OnProportions[:porIndex])
                    # 两个阶段的区间差距
                    gap = self.god_OnTimes[porIndex] - self.god_OnTimes[porIndex - 1]
                    # 用于迭代的 mu
                    mu = 1 / (self.__exppdf(gap, mu) * coef)
                    # 确定 x 坐标的上限
                    if porIndex == len(self.god_OnProportions) - 1:
                        xLimit = maxX + steps
                    else:
                        xLimit = self.god_OnTimes[porIndex+1]
                    # 添加 x 坐标和对应的 y 坐标
                    x, y = [], []
                    while pos < xLimit:
                        x.append(pos)
                        y.append(self.__exppdf(pos - self.god_OnTimes[porIndex], mu))
                        pos += steps
                    self.xList_PDF.append(np.array(x))
                    self.yList_PDF.append(np.array(y))
            # 若首次开机时间不为 0
            else:
                # 第一阶段概率密度为 0
                pos = 0
                x, y = [], []
                while pos < self.god_OnTimes[0]:
                    x.append(pos)
                    y.append(0.0)
                    pos += steps
                self.xList_PDF.append(np.array(x))
                self.yList_PDF.append(np.array(y))
                # 接下来区间的概率密度为偏置的指数分布
                x, y = [], []
                mu = 1 - self.god_OnTimes[0]
                while pos < self.god_OnTimes[1]:
                    x.append(pos)
                    y.append(self.__exppdf(pos - self.god_OnTimes[0], mu))
                    pos += steps
                self.xList_PDF.append(np.array(x))
                self.yList_PDF.append(np.array(y))
                # 接下来区间的概率密度为
                for porIndex in range(1, len(self.god_OnProportions)):
                    # 两个阶段的机器比例
                    coef = sum(self.god_OnProportions[:porIndex + 1]) / sum(self.god_OnProportions[:porIndex])
                    # 两个阶段的区间差距
                    gap = self.god_OnTimes[porIndex] - self.god_OnTimes[porIndex - 1]
                    # 用于迭代的 mu
                    mu = 1 / (self.__exppdf(gap, mu) * coef)
                    # 确定 x 坐标的上限
                    if porIndex == len(self.god_OnProportions) - 1:
                        xLimit = maxX + steps
                    else:
                        xLimit = self.god_OnTimes[porIndex + 1]
                    # 添加 x 坐标和对应的 y 坐标
                    x, y = [], []
                    while pos < xLimit:
                        x.append(pos)
                        y.append(self.__exppdf(pos - self.god_OnTimes[porIndex], mu))
                        pos += steps
                    self.xList_PDF.append(np.array(x))
                    self.yList_PDF.append(np.array(y))

        # 为使得原始概率密度函数构成的积分为 1，需要进行标准化处理（这就需要求积分）
        # 积分、指针
        integral, pos = 0, 0
        for line in self.yList_PDF:
            lineLength = len(line)
            for index in range(lineLength - 1):
                integral += (line[index] + line[index + 1]) * steps / 2
        # 根据积分重新修正概率密度函数
        for lineIndex in range(len(self.yList_PDF)):
            self.yList_PDF[lineIndex] /= integral
        # 打印最终得到的概率密度信息
        if printFlag:
            print("\n概率密度函数数据如下：")
            print("x 坐标（长度为 %d）："%(len(self.xList_PDF)))
            print(self.xList_PDF)
            print("y 坐标（长度为 %d）："%(len(self.yList_PDF)))
            print(self.yList_PDF)

    def __getCDF(self, printFlag = False):
        '''
        计算概率分布函数
        :return:
        '''
        # 积分、指针、步长
        integral, pos, steps = 0, 0, 0.001
        self.xList_CDF.append(0)
        self.yList_CDF.append(0)
        for line in self.yList_PDF:
            lineLength = len(line)
            for index in range(lineLength - 1):
                integral += (line[index] + line[index + 1]) * steps / 2
                pos += steps
                self.xList_CDF.append(pos)
                self.yList_CDF.append(integral)
            self.CDF_Stage.append(integral)
        self.xList_CDF = np.array(self.xList_CDF)
        self.yList_CDF = np.array(self.yList_CDF)
        if printFlag:
            print("\n开机节点处的分布函数")
            print(self.CDF_Stage)
            print("开机节点处的开启机器占比：")
            print("自私玩家：" + str(self.selfish_MachinePortions))
            print("诚实玩家：" + str(self.honest_MachinePortions))

    def drawPDF(self, maxX = 5):
        '''
        绘制概率密度函数
        :return:
        '''
        # 设置绘图板规格
        figWidth = 4
        plt.figure(figsize=(figWidth*1.618, figWidth))
        # 颜色数组
        colors = ["red", "darkorange", "gold", "limegreen", "turquoise", "blueviolet", "fuchsia", "deeppink"]
        # 获取概率密度中的最大 y 值
        maxY = self.yList_PDF[0][0]
        for line in self.yList_PDF:
            for value in line:
                maxY = max(maxY, value)
        # 绘制 x 轴
        plt.plot([0, maxX], [0, 0], c="black", linewidth=1)
        # 绘制 y 轴
        plt.plot([0, 0], [0, maxY], c="black", linewidth=1)
        # 绘图
        for lineIndex in range(len(self.xList_PDF)):
            # 对最后的一段数据需要根据指定 x 轴长度进行绘制
            if lineIndex == len(self.xList_PDF) - 1:
                x = self.xList_PDF[lineIndex][self.xList_PDF[lineIndex] <= maxX]
                y = self.yList_PDF[lineIndex][self.xList_PDF[lineIndex] <= maxX]
                plt.plot(x, y, c = colors[lineIndex])
            else:
                # 绘制该部分对应的概率密度函数
                plt.plot(self.xList_PDF[lineIndex], self.yList_PDF[lineIndex], c = colors[lineIndex])
                # 绘制用于分隔不同概率密度函数的虚线
                x = self.xList_PDF[lineIndex][-1]
                plt.plot([x, x], [0, maxY], ls = ":", c = "grey")
        plt.show()

    def drawCDF(self, maxX = 5):
        '''
        绘制概率分布函数
        :return:
        '''
        # 设置绘图板规格
        figWidth = 4
        plt.figure(figsize=(figWidth * 1.618, figWidth))
        # 颜色数组
        colors = ["red", "darkorange", "gold", "limegreen", "turquoise", "blueviolet", "fuchsia", "deeppink"]
        # 获取分布函数中的最大 y 值
        maxY = max(self.yList_CDF) + 0.05
        # 绘制 x 轴
        plt.plot([0, maxX], [0, 0], c="black", linewidth=1)
        # 绘制 y 轴
        plt.plot([0, 0], [0, maxY], c="black", linewidth=1)
        # 绘图
        x = self.xList_CDF[self.xList_CDF <= maxX]
        y = self.yList_CDF[self.xList_CDF <= maxX]
        plt.plot(x, y, c = "dodgerblue")
        # 绘制 y = 1 的渐近线
        plt.plot([0, maxX], [1, 1], c="black", ls=":")
        # 绘制均值 x = 1 的竖线
        plt.plot([1,1], [0, maxY], c="black", ls="--")
        plt.show()


    def __getProbability(self, printFlag = False):
        '''
        计算各时间节点处的转移概率
        :return:
        '''
        # 若首次开机时间不为 0
        if self.god_OnTimes[0] > 0.0001:
            self.selfish_Probability.append(self.selfish_MachinePortions[0] * self.CDF_Stage[1])
            self.honest_Probability.append(self.honest_MachinePortions[0] * self.CDF_Stage[1])
            for index in range(2, len(self.CDF_Stage) - 1):
                GAP = self.CDF_Stage[index] - self.CDF_Stage[index - 1]
                self.selfish_Probability.append(self.selfish_MachinePortions[index - 1] * GAP)
                self.honest_Probability.append(self.honest_MachinePortions[index - 1] * GAP)
            if len(self.CDF_Stage) > 2:
                GAP = 1 - self.CDF_Stage[-2]
                self.selfish_Probability.append(self.selfish_MachinePortions[-1] * GAP)
                self.honest_Probability.append(self.honest_MachinePortions[-1] * GAP)
        else:
            self.selfish_Probability.append(self.selfish_MachinePortions[0] * self.CDF_Stage[0])
            self.honest_Probability.append(self.honest_MachinePortions[0] * self.CDF_Stage[0])
            for index in range(1, len(self.CDF_Stage) - 1):
                GAP = self.CDF_Stage[index] - self.CDF_Stage[index - 1]
                self.selfish_Probability.append(self.selfish_MachinePortions[index] * GAP)
                self.honest_Probability.append(self.honest_MachinePortions[index] * GAP)
            if len(self.CDF_Stage) > 1:
                GAP = 1 - self.CDF_Stage[-2]
                self.selfish_Probability.append(self.selfish_MachinePortions[-1] * GAP)
                self.honest_Probability.append(self.honest_MachinePortions[-1] * GAP)
        self.alpha = sum(self.selfish_Probability)
        if printFlag:
            print("\n转移概率：")
            print("alpha=" + str(self.alpha) + "：" + str(self.selfish_Probability))
            print("beta=" + str(sum(self.honest_Probability)) + "：" + str(self.honest_Probability))

    def __getProbabilityGenerator(self, printFlag = False):
        '''
        概率生成器
        :return:
        '''
        extend = 1000
        # 遍历概率密度函数得到所有时间取值的频数
        for lineIndex in range(len(self.xList_PDF)):
            for index, p in enumerate(self.yList_PDF[lineIndex]):
                # 获取频数
                frequent = int(p * extend)
                for i in range(frequent):
                    self.probabilityBox.append(self.xList_PDF[lineIndex][index])
        # 置乱频数容器
        random.shuffle(self.probabilityBox)
        if printFlag:
            print("\n概率生成器内容：")
            print(self.probabilityBox)


    def getTransferProbability(self, printFlag = False):
        '''
        计算给定开机策略下，自私玩家和诚实玩家的转移概率
        :return:
        '''
        # 得到系统角度下的开机情况
        self.__getGodView(printFlag)
        # 计算概率密度函数
        self.__getPDF(printFlag)
        # 计算概率分布函数
        self.__getCDF(printFlag)
        # 计算转移概率
        self.__getProbability(printFlag)
        # 构建概率生成器
        self.__getProbabilityGenerator()


    def getNextTime(self):
        '''
        生成一个区块时对应的时间 T
        :return:
        '''
        return random.choice(self.probabilityBox)


    def __getRandomDirection(self, probability):
        '''
        根据转移概率执行一次随机跳转
        :param alpha:
        :return:
        '''
        if random.random() > probability:
            return 1
        else:
            return 0


    def __MarkovTransitions(self):
        '''
        模拟一次马尔可夫状态转移过程
        :param alpha: 诚实玩家的转移概率
        :param gamma: 权值
        :param STEPS: 单次模拟实验的步数
        :return:
        '''
        # 初态
        state = [0, 0]
        for i in range(self.steps):
            # 随机获取一个转移方向
            direct = self.__getRandomDirection(self.alpha)
            # 根据转移方向和当前状态决定下一步的转移状态
            if direct == 0:
                state[0] += 1
            else:
                # 遇到 beta 指向，且当前状态的第二个值为 0
                if state[1] == 0:
                    # 执行跳转
                    state[1] += 1
                    # 状态跳转检测
                    if state[0] <= state[1] or state[0] - state[1] == 1:
                        state = [0, 0]
                else:
                    # 遇到 beta 指向，且当前状态下第一个值比第二个值大 2
                    if state[0] - state[1] == 2:
                        # 执行跳转
                        state = [0, 0]
                    else:
                        # 需要根据 gamma 再获取一次随机方向
                        direct = self.__getRandomDirection(self.gamma)
                        # 执行跳转
                        if direct == 1:
                            state[direct] += 1
                        else:
                            state = [state[0] - state[1], 1]
            # 更新状态字典
            if tuple(state) in self.stateDict:
                self.stateDict[tuple(state)] = self.stateDict[tuple(state)] + 1
            else:
                self.stateDict[tuple(state)] = 1


    def getMarkovState(self, printFlag = False):
        '''
        进行指定次数的马尔可夫转移过程，并统计由这些实验得到状态概率
        :return:
        '''
        # 进行 expCount 次马尔可夫实验
        for i in range(self.markovExpCount):
            self.__MarkovTransitions()
        # 统计各状态的概率
        totalSteps = self.steps*self.markovExpCount
        for key in self.stateDict:
            self.stateDict[key] /= totalSteps
            if printFlag:
                print(str(key)+":"+str(self.stateDict[key]))
        SUM = 0
        for key in self.stateDict:
            SUM+=self.stateDict[key]


    def __getMeanBaseWardAndExpense(self):
        '''
         计算基本奖励和费用的平均值（基于指定次数）
        :return:
        '''
        selfishBaseReward = 0
        selfishExpense = 0
        honestBaseReward = 0
        honestExpense = 0
        averageT = 0
        for i in range(self.rewardExpCount):
            # 生成一个区块时对应的时间 T
            T = self.getNextTime()
            averageT += T
            # 计算基本奖励
            selfishBaseReward += self.__getSelfishBaseReward(T)
            honestBaseReward += self.__getHonestBaseReward(T)
            # 计算费用
            selfishExpense += self.__getSelfishExpense(T)
            honestExpense += self.__getHonestExpense(T)
        # 计算平均值
        self.__selfishBaseReward = selfishBaseReward / self.rewardExpCount
        self.__honestBaseReward = honestBaseReward / self.rewardExpCount
        self.__selfishExpense = selfishExpense / self.rewardExpCount
        self.__honestExpense = honestExpense / self.rewardExpCount
        averageT /= self.rewardExpCount
        # print("\n***************************************************************")
        # print("%d 次随机取数得到的 [下次区块生成时间] 均值为：%.4f" % (self.rewardExpCount, averageT))
        # print("***************************************************************\n")


    def rewardParameter(self, rewardExpCount, EBRR = 1, blockInterval = 1, lambda_t = 1, cop = 0.01, op = 0.01):
        '''
        奖励参数的设置（调用此函数后就能计算当前系统的基本奖励和开销）
        :param rewardExpCount: 计算基本奖励和开销的实验次数
        :param EBRR:
        :param blockInterval:
        :param lambda_t:
        :param cop:
        :param op:
        :return:
        '''
        # 比例控制系数
        self.EBRR = EBRR
        # 预期的区块间隔时间
        self.blockInterval = blockInterval
        # 奖励的偏置系数
        self.lambda_0 = EBRR * blockInterval * lambda_t
        # 奖励随时间的比例系数
        self.lambda_t = lambda_t
        # 机器固定费用
        self.C_cop = cop
        # 开机后产生的费用
        self.C_op = op
        # 计算基本奖励和开销的实验次数
        self.rewardExpCount = rewardExpCount
        # 计算基本奖励和开销
        self.__getMeanBaseWardAndExpense()


    def __getRu(self, d):
        '''
        以太坊中的 Ru 函数
        :param d:
        :return:
        '''
        if 1<= d and d <= 6:
            return (8-d)/8
        else:
            return 0

    def __getRn(self, n):
        '''
        以太坊中的 Rn 函数
        :param n:
        :return:
        '''
        return n*self.lambda_0/32


    def __getSelfishBaseReward(self, T):
        '''
        计算自私玩家的基本奖励（这个奖励依赖于时间）
        :return:
        '''
        reward = 0
        alpha = self.alpha
        beta = 1 - alpha
        gamma = self.gamma
        # 计算在各状态上的基本奖励
        for state in self.stateDict:
            if state == (0, 0):
                reward += self.stateDict[state] * (alpha * alpha + alpha * alpha * beta + alpha * beta * beta * gamma)
            else:
                reward += self.stateDict[state] * alpha
        return reward * (self.lambda_0 + self.lambda_t * T)


    def __getHonestBaseReward(self, T):
        '''
        计算诚实玩家的基本奖励（这个奖励依赖于时间）
        :return:
        '''
        reward = 0
        beta = 1 - self.alpha
        # 计算状态 (0,0) 的基本奖励
        if (0, 0) in self.stateDict:
            reward += self.stateDict[(0, 0)] * beta
        # 计算状态 (1,1) 的基本奖励
        if (1, 1) in self.stateDict:
            reward += self.stateDict[(1, 1)] * beta
        # 计算状态 (1,0) 的基本奖励
        if (1, 0) in self.stateDict:
            reward += self.stateDict[(1, 0)] * beta * beta * (1 - self.gamma)
        return reward * (self.lambda_0 + self.lambda_t * T)


    def __getSelfishUncleReward(self):
        '''
        计算自私玩家的叔叔奖励
        :return:
        '''
        reward = 0
        alpha = self.alpha
        beta = 1 - alpha
        gamma = self.gamma
        if (0,0) in self.stateDict:
            reward += self.stateDict[(0,0)]*alpha*beta*beta*(1-gamma)*self.__getRu(1)
        else:
            reward = 0
        return reward


    def __getHonestUncleReward(self):
        '''
        计算诚实玩家的叔叔奖励
        :return:
        '''
        reward = 0
        alpha = self.alpha
        beta = 1 - alpha
        gamma = self.gamma
        for state in self.stateDict:
            if state == (1,0):
                reward += self.stateDict[state]*(alpha*beta + beta*beta*gamma)*self.__getRu(1)
            elif state[1] == 0 and state[0] >= 2:
                reward += self.stateDict[state]*beta*self.__getRu(state[0])
            elif state[0] >= 3 and state[1] >= 1:
                reward += self.stateDict[state]*beta*gamma*self.__getRu(state[0])
        return reward


    def __getSelfishNephewReward(self):
        '''
        计算自私玩家的侄子奖励
        :return:
        '''
        reward = 0
        alpha = self.alpha
        beta = 1 - alpha
        gamma = self.gamma
        for state in self.stateDict:
            if state == (1,1):
                reward += self.stateDict[(1,1)]*alpha*self.__getRn(1)
            elif state == (3, 1):
                reward += self.stateDict[(3, 1)] * beta * gamma *(alpha - alpha*beta*beta*(1-gamma))*self.__getRn(2)
            elif state[0] >= 4 and state[1] >= 1:
                reward += self.stateDict[state]*beta*gamma*pow(beta*gamma,state[0]-state[1]-2)*(alpha-alpha*beta*beta*(1-gamma))*self.__getRn(state[0]-state[1])
        return reward


    def __getHonestNephewReward(self):
        '''
        计算诚实玩家的侄子奖励
        :return:
        '''
        reward = 0
        alpha = self.alpha
        beta = 1 - alpha
        gamma = self.gamma
        for state in self.stateDict:
            if state == (1, 1):
                reward += self.stateDict[(1, 1)] * beta * self.__getRn(1)
            elif state == (3, 1):
                reward += self.stateDict[(3, 1)] * beta * gamma * (
                            beta + alpha * beta * beta * (1 - gamma)) * self.__getRn(2)
            elif state[0] >= 4 and state[1] >= 1:
                reward += self.stateDict[state] * beta * gamma * pow(beta * gamma, state[0] - state[1] - 2) * (
                                beta + alpha * beta * beta * (1 - gamma)) * self.__getRn(state[0] - state[1])
        return reward


    def __getSelfishExpense(self, T):
        '''
        计算自私玩家的预期总费用（这个奖励依赖于时间）
        :return:
        '''
        TimeGap = 0
        for t in self.selfish_OnTimes:
            TimeGap += (T - t)
        return self.C_cop * self.selfish_Total * T + self.C_op * TimeGap


    def __getHonestExpense(self, T):
        '''
        计算诚实玩家的预期总费用（这个奖励依赖于时间）
        :return:
        '''
        TimeGap = 0
        for t in self.honest_OnTimes:
            TimeGap += (T - t)
        return self.C_cop * self.honest_Total * T + self.C_op * TimeGap


    def getUtility(self, printFlag = False):
        '''
        计算效用
        :return:
        '''
        # 自私玩家
        selfishBaseReward = self.__selfishBaseReward
        selfishUncleReward = self.__getSelfishUncleReward()
        selfishNephewReward = self.__getSelfishNephewReward()
        selfishExpense = self.__selfishExpense
        if printFlag:
            print("---------------------------")
            print("自私玩家的基本奖励：%.6f" % selfishBaseReward)
            print("自私玩家的叔叔奖励：%.6f" % selfishUncleReward)
            print("自私玩家的侄子奖励：%.6f" % selfishNephewReward)
            print("自私玩家的基本开销：%.6f" % selfishExpense)
        selfishUtility = selfishBaseReward + selfishUncleReward + selfishNephewReward - selfishExpense
        # 诚实玩家
        honestBaseReward = self.__honestBaseReward
        honestUncleReward = self.__getHonestUncleReward()
        honestNephewReward = self.__getHonestNephewReward()
        honestExpense = self.__honestExpense
        if printFlag:
            print("----")
            print("诚实玩家的基本奖励：%.6f" % honestBaseReward)
            print("诚实玩家的叔叔奖励：%.6f" % honestUncleReward)
            print("诚实玩家的侄子奖励：%.6f" % honestNephewReward)
            print("诚实玩家的基本开销：%.6f" % honestExpense)
            print("---------------------------")
        honestUtility = honestBaseReward + honestUncleReward + honestNephewReward - honestExpense
        return selfishUtility, honestUtility


def findBestOntimes(playerClass, selfMachineTotal, objectOntime, objectMachineTotal, EBRR, C_op, C_cop):
    '''
    根据现有开机策略寻找此刻的最佳开机时间
    :param playerClass: 角色类型
    :param selfMachineTotal: 角色总机器数
    :param objectOntimeList: 其他角色的开机时间序列
    :param objectOnProportionList: 其他角色的开机比例序列
    :param objectMachineTotal: 其他角色的总机器数
    :return:
    '''
    # SelfishMining 相关参数
    gamma, steps, limit, ebrr, op, cop = 0.5, 200, 100000, EBRR, C_op, C_cop

    # 设置玩家的开机时间序列
    PlayerOntimes = np.linspace(0, 1, 101)[:-1]
    # 玩家的最优开机时间
    bestOntime = -1
    # 玩家的最高效用
    bestUtility = -10000
    # 寻找玩家的最优开启时间
    if playerClass == 0:
        for PlayerOntime in PlayerOntimes:
            objSelfish = SelfishMining([PlayerOntime], [1], selfMachineTotal,
                                       [objectOntime], [1], objectMachineTotal,
                                       gamma, steps, limit)
            # 计算转移概率（这一步已经实现了玩家的开机序列融合）
            objSelfish.getTransferProbability()
            # 进行马尔可夫状态转移
            objSelfish.getMarkovState()
            # 设置奖励参数
            objSelfish.rewardParameter(rewardExpCount=10000, EBRR=ebrr, op=op, cop=cop)
            # 计算绝对效用
            selfishUtility, honestUtility = objSelfish.getUtility()
            # 记录最优效用及对应的开机时间
            if selfishUtility > bestUtility:
                bestUtility = selfishUtility
                bestOntime = PlayerOntime
    else:
        for PlayerOntime in PlayerOntimes:
            objHonest = SelfishMining([objectOntime], [1], objectMachineTotal,
                                       [PlayerOntime], [1], selfMachineTotal,
                                       gamma, steps, limit)
            # 计算转移概率（这一步已经实现了玩家的开机序列融合）
            objHonest.getTransferProbability()
            # 进行马尔可夫状态转移
            objHonest.getMarkovState()
            # 设置奖励参数
            objHonest.rewardParameter(rewardExpCount=10000, EBRR=ebrr, op=op, cop=cop)
            # 计算绝对效用
            selfishUtility, honestUtility = objHonest.getUtility()
            # 记录最优效用及对应的开机时间
            if honestUtility > bestUtility:
                bestUtility = honestUtility
                bestOntime = PlayerOntime
    return bestOntime


def getResult(EBRR, C_op, C_cop, savePath, machineTotal = 256, ConvergenceThreshold = 0.02):
    '''
    自私玩家以及诚实玩家的最优开启时间
    :param C_op:
    :param C_cop:
    :param savePath:
    :param machineTotal:
    :return:
    '''
    # 自私玩家的相对算力
    selfishPortions = np.linspace(0.1, 0.5, 9)
    # 设置随机数种子（保证后续实验总是用的同一数据）
    random.seed(6)

    for selfishPortion in selfishPortions:
        print("比例为 %.1f 时：" % selfishPortion)
        # 全部玩家的机器总数序列
        PlayerMachineTotal = [machineTotal * selfishPortion, machineTotal - machineTotal * selfishPortion]
        # 全部玩家的开机时间序列
        PlayerOntimes = [[], [random.randint(1, 100) / 100]]

        cnt = 1
        # 迭代查找各自的最优开机时间
        while True:
            print("  第 %d 次迭代" % cnt)
            # 计算指定角色的最优开机时间
            for roleIndex in range(2):
                # 寻找指定角色的最优开机时间
                bestOntime = findBestOntimes(roleIndex, PlayerMachineTotal[roleIndex], PlayerOntimes[1-roleIndex][-1],
                                              PlayerMachineTotal[1-roleIndex], EBRR, C_op, C_cop)
                # 为当前角色添加最优开机时间
                PlayerOntimes[roleIndex].append(bestOntime)
            print("  玩家 1：" + str(PlayerOntimes[0]))
            print("  玩家 2：" + str(PlayerOntimes[1]))
            cnt += 1
            # 判断两个角色的开机时间是否收敛
            if len(PlayerOntimes[0]) > 1 and len(PlayerOntimes[1]) > 1:
                if (abs(PlayerOntimes[0][-1] - PlayerOntimes[0][-2]) <= ConvergenceThreshold) and \
                        (abs(PlayerOntimes[1][-1] - PlayerOntimes[1][-2]) <= ConvergenceThreshold):
                            break
            # 如果迭代次数达到 20 则认为是震荡收敛，终止程序
            if cnt == 20:
                break
        # 记录实验数据
        with open(savePath, "a") as f:
            f.write("Relative mining power of selfish players:" + str(selfishPortion) + "\n")
            for line in PlayerOntimes:
                lineLen = len(line)
                for index, onTime in enumerate(line):
                    if index != lineLen - 1:
                        f.write(str(onTime)+",")
                    else:
                        f.write(str(onTime)+"\n")

if __name__ == '__main__':
    # 实验 4
    # 并行计算
    print("图 3: EBRR = 0")
    getResult(0, 0, 0.02, "../ExperimentData/exp4_3_EBRR0.csv")
    print("图 3: EBRR = 0.5")
    getResult(0.5, 0, 0.02, "../ExperimentData/exp4_3_EBRR1.csv")
    print("图 3: EBRR = 1")
    getResult(1, 0, 0.02, "../ExperimentData/exp4_3_EBRR2.csv")
    print("图 3: EBRR = 1.5")
    getResult(1.5, 0, 0.02, "../ExperimentData/exp4_3_EBRR3.csv")
    print("图 3: EBRR = 2")
    getResult(2, 0, 0.02, "../ExperimentData/exp4_3_EBRR4.csv")

