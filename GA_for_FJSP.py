import numpy as np
import random
from Decode_for_FJSP import Decode
from Encode_for_FJSP import Encode
import itertools
import matplotlib.pyplot as plt

class GA:
    def __init__(self):
        # self.Pop_size = 200       #种群数量
        self.Pop_size = 500  # 种群数量
        self.P_c = 0.8            #交叉概率
        # self.P_m=0.3            #变异概率
        self.P_m = 0.03
        self.P_v = 0.5            #选择何种方式进行交叉
        # self.P_w=0.99            #采用何种方式进行变异
        self.P_w = 0.6
        # self.Max_Itertions = 100  #最大迭代次数
        self.Max_Itertions = 200

    #适应度
    def fitness(self,CHS,J,Processing_time,M_num,Len):
        Fit1=[]
        for i in range(len(CHS)):
            d = Decode(J, Processing_time, M_num)
            Fit1.append(d.Decode_1(CHS[i],Len))         # d.Decode_1(CHS[i],Len) : 返回了这个适应度——不过这里是，完成了所有工序，最后一道工序的完工时间。
        Fit2 = []
        for i in range(len(CHS)):
            d = Decode(J, Processing_time, M_num)
            Fit2.append(d.Decode_2(CHS[i], Len))
        # 这里返回了两部分 fit 的结果值；目前来看，除了一些特殊的时间窗口，这两个应该是相等的。
        # return Fit1,Fit2                              # # 2023.04.13 修改，不需要对比两个 Fit
        return Fit1

    #机器部分交叉
    def Crossover_Machine(self,CHS1,CHS2,T0):
        """

        :param CHS1: 个体 1
        :param CHS2: 个体 2
        :param T0:  工序数量
        :return:
        """
        T_r=[j for j in range(T0)]      # 总共有这么多道工序
        r = random.randint(1, 10)       # 在区间[1,T0]内产生一个整数r
        random.shuffle(T_r)
        R = T_r[0:r]  # 按照随机数r产生r个互不相等的整数
        # 将父代的染色体复制到子代中去，保持他们的顺序和位置
        OS_1 = CHS1[T0:2 * T0]      # 工序数量后面的， 其实是工序的顺序（打乱后的）
        OS_2 = CHS2[T0:2 * T0]      # 同上
        C_1 = CHS2[0:T0]            #
        C_2 = CHS1[0:T0]
        for i in R:                 # 互换 C_1、 C_2  里面 R 个元素的索引对印的数
            K,K_2 = C_1[i],C_2[i]
            C_1[i],C_2[i] = K_2,K
        CHS1 = np.hstack((C_1,OS_1))
        CHS2 = np.hstack((C_2, OS_2))
        return CHS1,CHS2

    #工序交叉部分
    def Crossover_Operation(self,CHS1, CHS2, T0, J_num):
        # T0 ： 工序数量
        OS_1 = CHS1[T0:2 * T0]
        OS_2 = CHS2[T0:2 * T0]
        MS_1 =CHS1[0:T0]
        MS_2 = CHS2[0:T0]
        Job_list = [i for i in range(J_num)]
        random.shuffle(Job_list)
        r = random.randint(1, J_num - 1)
        Set1 = Job_list[0:r]
        Set2 = Job_list[r:J_num]
        new_os = list(np.zeros(T0, dtype=int))
        for k, v in enumerate(OS_1):
            if v in Set1:
                new_os[k] = v + 1
        for i in OS_2:
            if i not in Set1:
                Site = new_os.index(0)      # 这里，每次返回列表里面为 0 的位置的索引
                new_os[Site] = i + 1        # i + 1 赋值给 new_os 为 0  的点
        new_os = np.array([j - 1 for j in new_os])          # 前面 + 1，这里再减1..
        CHS1 = np.hstack((MS_1,new_os))                     # 横向拼接
        CHS2 = np.hstack((MS_2, new_os))
        return CHS1,CHS2

    def reduction(self,num,J,T0):
        """

        :param num:  第几道工序
        :param J:    每个工件的工序数
        :param T0:   总工序数
        :return:     工件、 这道工序在 该工件里面的加工顺序
        """
        T0=[j for j in range(T0)]
        K=[]
        Site=0
        for k,v in J.items():
            K.append(T0[Site:Site+v])
            Site+=v
        for i in range(len(K)):         # 有多少个工件， len(K) 就有多长
            if num in K[i]:
                Job = i                 # 工件
                O_num = K[i].index(num)     # 这道工序在 工件 K 里面的加工顺序
                break
        return Job,O_num

    #机器变异部分
    def Variation_Machine(self,CHS,O,T0,J):
        """

        :param CHS:  个体
        :param O:    每个工件，每道工序、在每台机器上的加工时间
        :param T0:   工序数量
        :param J:   工件、工序数
        :return:    机器变异后的新的个体
        """
        Tr = [i_num for i_num in range(T0)]
        MS = CHS[0:T0]
        OS = CHS[T0:2*T0]
        # 机器选择部分
        r = random.randint(1, T0 - 1)  # 在变异染色体中选择r个位置
        random.shuffle(Tr)
        T_r = Tr[0:r]
        for i in T_r:
            Job = self.reduction(i,J,T0)
            O_i = Job[0]        # 工件
            O_j = Job[1]        # 对应工件的第几道加工顺序
            Machine_using = O[O_i][O_j]         # 可以选用的机器
            Machine_time = []
            for j in Machine_using:
                if j != 9999:
                    Machine_time.append(j)
            Min_index = Machine_time.index(min(Machine_time))
            MS[i] = Min_index               # 所以说，这里的机器变异，就是把这道工序对应的机器， 替换成了 加工这道工序时间最短的那台机器，进行加工。
        CHS=np.hstack((MS,OS))
        return CHS
    #工序变异部分
    def Variation_Operation(self, CHS,T0,J_num,J,Processing_time,M_num):
        """

        :param CHS: 个体
        :param T0:  工序数量
        :param J_num: 工件数量
        :param J:       工件及其工序信息
        :param Processing_time:     加工时长
        :param M_num:   机器数量
        :return:
        """
        MS = CHS[0:T0]
        OS = list(CHS[T0:2*T0])
        r = random.randint(1,J_num-1)
        Tr = [i for i in range(J_num)]
        random.shuffle(Tr)
        Tr = Tr[0:r]
        J_os = dict(enumerate(OS))    #随机选择r个不同的基因
        J_os = sorted(J_os.items(), key=lambda d: d[1])
        Site = []                   # 储存 Tr 里面每个元素的索引
        for i in range(r):
            Site.append(OS.index(Tr[i]))
        A = list(itertools.permutations(Tr, r))     # Tr 的全排列，全都放了出来。   # TODO  这里全排列，太费时间了。。。
        if len(A) <= 200:
            A_CHS = []
            for i in range(len(A)):                     # 全排列产生了多少个个体，这里 A_CHS 就记录了多少个个体
                for j in range(len(A[i])):
                    OS[Site[j]]=A[i][j]                 # 工序部分的 OS ，其实在这里已经改变了。
                C_I=np.hstack((MS,OS))
                A_CHS.append(C_I)
            Fit = []
            for i in range(len(A_CHS)):
                # TODO 这里应该要修改成 A_CHS 里面的 CHS才对
                CHS = A_CHS[i]
                d = Decode(J, Processing_time, M_num)
                Fit.append(d.Decode_1(CHS, T0))
            return A_CHS[Fit.index(min(Fit))]
        else:
            return CHS

    def Select(self,Fit_value):
        Fit=[]
        for i in range(len(Fit_value)):
            fit = 1 / Fit_value[i]          # Fit_value[i] 是完工时间， 取倒数，才是取适应度最大的数。
            Fit.append(fit)
        Fit = np.array(Fit)                 # 这里存的是真实的 适应度。
        # 依照每个个体的适应度，重新选择完整的 Fit； replace=True 表示适应度高的个体，可以重复选择。 这里重复选择的，其实是适应度高的 index
        idx = np.random.choice(np.arange(len(Fit_value)), size=len(Fit_value), replace=True, p= (Fit) / (Fit.sum())  )
        return idx

    def main(self,Processing_time,J,M_num,J_num,O_num):
        e = Encode(Processing_time, self.Pop_size, J, J_num, M_num)
        OS_List = e.OS_List()
        Len_Chromo = e.Len_Chromo         # 所有工件的所有工序数总和
        CHS1 = e.Global_initial()
        CHS2 = e.Random_initial()
        CHS3 = e.Local_initial()
        C=np.vstack((CHS1,CHS2,CHS3))
        Optimal_fit=9999
        Optimal_CHS=0
        # x = np.linspace(0, 30, 30)
        x = np.linspace(0, self.Max_Itertions, self.Max_Itertions)
        x1=[x1 for x1 in range(self.Pop_size)]
        Best_fit=[]
        for i in range(self.Max_Itertions):
            Fit = self.fitness(C, J, Processing_time, M_num, Len_Chromo)
            # plt.plot(x1,Fit[0],'-k')                      # 2023.04.13 修改，不需要对比两个 Fit
            # plt.plot(x1, Fit[1], linestyle='-.',c="black")

            # plt.plot(x1, Fit, '-k')
            # plt.title('Comparision between two scheduling strategies')
            # plt.ylabel('Cmax')
            # plt.xlabel('Individual')
            # plt.show()

            # Fit = Fit[0]                        # 2023.04.13 更新
            min_fitness_idx = np.argmin(Fit)
            Best = C[min_fitness_idx]           # 这个是最好的个体的索引
            # best_fitness = min(Fit)
            best_fitness = Fit[min_fitness_idx]
            print(f"\n----------------------当前第{i}次迭代，最好的结果是在 {Optimal_fit} 时间内完成所有工件所有工序加工-------------\n  ")
            if best_fitness < Optimal_fit:
                Optimal_fit = best_fitness
                Optimal_CHS = Best
                Best_fit.append(Optimal_fit)
                print('best_fitness', best_fitness)
                d = Decode(J, Processing_time, M_num)
                Fit.append(d.Decode_1(Optimal_CHS, Len_Chromo))                 # 已经找到适应度最好的个体， 应该不需要再填充一次这个适应度到  Fit 中去吧。
                max_fit_idx = np.argmax(Fit)                                    # 添加了一个个体，再删除一个当前里面最不好的个体
                Fit.pop(max_fit_idx)
                d.Gantt(d.Machines)                         # 画甘特图
            else:
                Best_fit.append(Optimal_fit)
            Select = self.Select(Fit)
            # C=[C[Select_i-1] for Select_i in Select]              # 这里重新刷新了 C ； 即：在这里依据个体的适应度大小，刷新了整个种群的数据，优先选取了适应度高的个体存留了下来。
            C = [C[Select_i] for Select_i in Select]                #  不要减一， 都是索引，不用减一
            for j in range(len(C)):
                offspring = []
                if random.random()<self.P_c:                        # 交叉概率
                    N_i = random.choice(np.arange(len(C)))
                    if random.random() < self.P_v:                  # 选择何种方式进行交叉
                        # 机器部分交叉
                        # 选择 当前个体、 然后随机选择一个个体，两个个体之间考虑是否交叉
                        Crossover = self.Crossover_Machine(C[j],C[N_i],Len_Chromo)
                        # print('Cov1----->>>>>',len(Crossover[0]),len(Crossover[1]))
                    else:
                        # 工序部分交叉
                        Crossover=self.Crossover_Operation(C[j],C[N_i],Len_Chromo,J_num)        # J_num ： 工件数量

                    # 交叉后，返回两个新的个体
                    offspring.append(Crossover[0])
                    offspring.append(Crossover[1])

                if random.random()<self.P_m:                # 变异概率
                    if random.random() < self.P_w:
                        # 机器变异
                        Mutation = self.Variation_Machine(C[j],Processing_time,Len_Chromo,J)
                    else:
                        # 工序变异
                        Mutation = self.Variation_Operation(C[j],Len_Chromo,J_num,J,Processing_time,M_num)
                    offspring.append(Mutation)
                if offspring !=[]:
                    # Fit = []
                    # for i in range(len(offspring)):
                    #     d = Decode(J, Processing_time, M_num)
                    #     Fit.append(d.Decode_1(offspring[i], Len_Chromo))
                    C[j] = random.choice(offspring)         # 不一定会选哪一个，随机选一个作为新的 C[j]
        plt.plot(x, Best_fit,'-k')
        plt.title(
            'the maximum completion time of each iteration for flexible job shop scheduling problem')
        plt.ylabel('Cmax')
        plt.xlabel('Test Num')
        plt.show()

if __name__=='__main__':
    from MK01 import Processing_time, J, M_num, J_num, O_num
    g=GA()
    g.main(Processing_time,J,M_num,J_num,O_num)








