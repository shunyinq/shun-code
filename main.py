import pandas as pd
import numpy as np
import time  ## time
# 生成了item和itemlist
import random
import pandas as pd
import numpy as np
import time     ## time
#生成了item和itemlist
import random


FFmaxfit=[]
FFtime=[]
GAFFmaxfit=[]
GAFFtime=[]

for sumitem in range(10,101):
    item = []
    a = 0
    b = 0
    i = 0
    while i < sumitem:  # this is the Generateed items, and 100 is the number of the items generated
        a = random.randint(1, 100)
        item.append(a)
        a = 0
        i = i + 1
    # print(item)
    crossitem = item

    # %%
    # 完成FF的函数
    # First Fit algorithm.

    def firstFit(weight, n, c):
        # Initialize result (Count of bins)     bin的计数器？？？？？？,,对的，就是
        res = 0  # res就是用的箱子数量

        # Create an array （bin_rem）  to store remaining space in bins
        # there can be at most n bins  新概念，最多n个 bin
        bin_rem = [0] * n  # 初始化，暂且全置0  但其实是剩余的空间
        # 初始化解，是二维数组(solution)，，一个箱子一个空数组，append加东西，，weight也还是前面的item
        solution = [[] for s in range(n)]
        Chromosome = []
        for i in range(n):  # n是箱子总数，初始化和物品数量一样的箱子数来面对最糟情况
            # Find the first bin that can accommodate   找到第一个能容纳的桶
            # weight[i]    就是第i个的大小
            j = 0
            while (j < res):  # 在已经用了的箱子里面一个个遍历        这个是双指针结构
                if (bin_rem[j] >= weight[i]):
                    bin_rem[j] = bin_rem[j] - weight[i]
                    solution[j].append(i + 1)
                    Chromosome.append(j + 1)
                    break
                j += 1
            # If no bin could accommodate weight[i]
            if (j == res):  # 开新的箱子了
                bin_rem[res] = c - weight[i]
                solution[res].append(i + 1)
                Chromosome.append(res + 1)
                res = res + 1

        solution = [x for x in solution if x]
        return res, Chromosome, solution


    ## res是所需箱子数， Chromosome是单个染色体， solution是一个箱子内部的内容

    # start=.timetime()         ## time

    # %%
    # popsize是问题规模，打乱几次item的顺序
    # pops是染色体构成的二维数组
    # VSUM 是每个箱子的占有空间的大小，二维数组
    # fit是每个解的fit值构成的一维数组
    # 以上的几个列表全都严格的一一对应
    FFstart = time.time()  ## time
    pops = []
    VSUM = []
    popsize = 100
    itemlist = [random.sample(item, len(item)) for j in range(popsize)]
    n = len(item)  # n是箱子总数，初始化和物品数量一样的箱子数来面对最糟情况
    c = 100
    boxNumlist = []

    for i in range(popsize):  # 依次循环出的解，各种形式都要存放， itemlist(每一次的item详情，二维数组）, 对应的chrom（二维数组pops）, solution（三维数组吧）等等

        weight = itemlist[i]
        num, chrom, solut = firstFit(weight, n, c)
        pops.append(chrom)
        num = list(range(len(weight)))
        cargo_df = pd.DataFrame({'itemnumber': num, "weight": weight})  # 简单一步就完成了物品从list 转换成pandas
        boxNum = max(chrom)  # 是数组中最大数不加（第几号箱子）
        boxNumlist.append(boxNum)
        boxes = [[] for a in range(boxNum)]  # 这个 boxNum就是有几个箱子就是几
        v_sum = [0] * boxNum  # 计算每一个箱子的载重        目前正确
        for j in range(len(chrom)):
            box_id = int(chrom[j]) - 1
            v_j = cargo_df[cargo_df['itemnumber'] == j]['weight'].iloc[0]  # 提取j号物品的体积
            boxes[box_id].append(j)  # 二维数组， boxes里面的box_id位箱子（原本是[])， 装进了 j，j对应的是物品号码，这个就是我FF的solution
            v_sum[box_id] += v_j
        # fit[i] = 100 / (np.var(v_sum))
        VSUM.append(v_sum)

    # %%
    fit = []

    c = 100
    n = 10
    for i in range(popsize):

        b = sum(VSUM[i]) / c / boxNumlist[i]


        fit.append(b)

        # %%

    # print(fit)

    # %%
    max_fit = max(fit)  # 返回最大值
    max_index = fit.index(max(fit))  # 最大值的索引
    bestFFChromosome = pops[max_index]
    bestitemlist = itemlist[max_index]

    FFmaxfit.append(max_fit)
    # print(max_fit)  # 最大的fit值
    # print(bestFFChromosome)  # 最好的染色体
    # print(bestitemlist)  # 最好的装箱顺序

    FFend = time.time()  ## time
    FFT=FFend - FFstart  ## time
    FFtime.append(FFT)
    # print(VSUM)
    # print(max_index)
    # print(itemlist)
    # print(pops)




#%%

    # %%

    # 完成FF的函数
    # First Fit algorithm.

    def firstFit(weight, n, c):
        # Initialize result (Count of bins)     bin的计数器？？？？？？,,对的，就是
        res = 0  # res就是用的箱子数量

        # Create an array （bin_rem）  to store remaining space in bins
        # there can be at most n bins  新概念，最多n个 bin
        bin_rem = [0] * n  # 初始化，暂且全置0  但其实是剩余的空间
        # 初始化解，是二维数组(solution)，，一个箱子一个空数组，append加东西，，weight也还是前面的item
        solution = [[] for s in range(n)]

        yuanlairanseti = []
        Chromosome = []
        for i in range(n):  # n是箱子总数，初始化和物品数量一样的箱子数来面对最糟情况
            # Find the first bin that can accommodate   找到第一个能容纳的桶
            # weight[i]    就是第i个的大小
            j = 0
            while (j < res):  # 在已经用了的箱子里面一个个遍历        这个是双指针结构
                if (bin_rem[j] >= weight[i]):
                    bin_rem[j] = bin_rem[j] - weight[i]  # 这个就是，装着的货物就是weight[i]
                    solution[j].append(i + 1)
                    yuanlairanseti.append(j + 1)  # 这个只能确认是箱子号，箱子号就是 j+1, 我现在需要第 n 个物品

                    ddd = []
                    eee = []
                    ccc = []
                    ddd.append(weight[i])
                    eee.append(j + 1)
                    ccc.append(ddd)
                    ccc.append(eee)
                    Chromosome.append(ccc)
                    ddd = []
                    eee = []
                    ccc = []

                    break
                j += 1
            # If no bin could accommodate weight[i]
            if (j == res):  # 开新的箱子了
                bin_rem[res] = c - weight[i]  # 这就是第 i 个物品，是从编号 0 开始算的，所以weight = weight[i]
                solution[res].append(i + 1)
                yuanlairanseti.append(res + 1)  # 箱子 = res + 1
                ddd = []
                eee = []
                ccc = []
                ddd.append(weight[i])
                eee.append(res + 1)
                ccc.append(ddd)
                ccc.append(eee)
                Chromosome.append(ccc)
                ddd = []
                eee = []
                ccc = []

                res = res + 1

        solution = [x for x in solution if x]
        return res, Chromosome, solution, yuanlairanseti


    ## res是所需箱子数， Chromosome是单个染色体， solution是一个箱子内部的内容

    # start=.timetime()         ## time



    # %%
    # popsize是问题规模，打乱几次item的顺序
    # pops是染色体构成的二维数组
    # VSUM 是每个箱子的占有空间的大小，二维数组
    # fit是每个解的fit值构成的一维数组
    # 以上的几个列表全都严格的一一对应
    start = time.time()  ## time
    pops = []
    VSUM = []
    popsizee = 100
    itemlist = [random.sample(item, len(item)) for j in range(popsizee)]
    n = len(item)  # n是箱子总数，初始化和物品数量一样的箱子数来面对最糟情况
    c = 100
    for i in range(popsizee):  # 依次循环出的解，各种形式都要存放， itemlist(每一次的item详情，二维数组）, 对应的chrom（二维数组pops）, solution（三维数组吧）等等

        weight = itemlist[i]
        num, chrom, solut, yuanlai = firstFit(weight, n, c)
        pops.append(chrom)  # 新的染色体，为后面的遗传算法铺垫,,后面不用 df ，就用这个自带的重量标签
        num = list(range(len(weight)))
        cargo_df = pd.DataFrame({'货物序号': num, "体积": weight})  # 简单一步就完成了物品从list 转换成pandas

        boxNum = max(yuanlai)  # 是数组中最大数不加（第几号箱子），，Chrome改成yuanlai
        boxes = [[] for a in range(boxNum)]  # 这个 boxNum就是有几个箱子就是几
        v_sum = [0] * boxNum  # 计算每一个箱子的载重        目前正确
        for j in range(len(yuanlai)):
            box_id = int(yuanlai[j]) - 1
            v_j = cargo_df[cargo_df['货物序号'] == j]['体积'].iloc[0]  # 提取j号物品的体积
            boxes[box_id].append(j)  # 二维数组， boxes里面的box_id位箱子（原本是[])， 装进了 j，j对应的是物品号码，这个就是我FF的solution
            v_sum[box_id] += v_j
        # fit[i] = 100 / (np.var(v_sum))
        VSUM.append(v_sum)

    # %%
    fit = []
    a = 0
    for i in range(popsizee):
        a = 100 / (np.var(VSUM[i]))  #
        fit.append(a)
    max_fit = max(fit)  # 返回最大值
    max_index = fit.index(max(fit))  # 最大值的索引
    bestFFChromosome = pops[max_index]
    bestitemlist = itemlist[max_index]
    # print(max_fit)  # 最大的fit值
    # print(bestFFChromosome)  # 最好的染色体
    # print(bestitemlist)  # 最好的装箱顺序

    # end = time.time()  ## time
    # print('Running time: %s Seconds' % (end - start))  ## time

    # 注意，在FF代码中，popsize就是指单纯的生成的shuffle后的item的数量，然后，，也是popss解（后文的染色体）的数量。
    # 。。这个得放进ga里面，得想明白种群生成的问题

    # print(pops)  # 初始种群
    #
    # # #%%
    # print(len(pops[4]))


    # %%

    # 将boxnumlist改为方法

    def calculate_boxnumlist(pops):

        chr = []

        for i in range(len(pops)):  # 针对popsize个染色体

            c = pops[i]

            nn = []
            for ii in range(len(c)):  # 针对一个染色体,提取每个元素的箱子号
                bb = c[ii]
                cc = bb[1]  # cc就是数字了
                rr = cc[0]
                nn.append(rr)
                bb = []

            chr.append(nn)

        # print(chr)

        # print(len(boxnumlist))

        boxnumlist = []
        for i in range(len(chr)):
            c = max(chr[i])
            boxnumlist.append(c)
            c = []

        return boxnumlist


    boxnumlist = calculate_boxnumlist(pops)
    # print(boxnumlist)


    # %%

    # # 生成boxnumlist
    #
    # chr=[]
    #
    #
    # for i in range(len(pops)):    # 针对popsize个染色体
    #
    #     c=pops[i]
    #
    #     nn=[]
    #     for ii in range(len(c)):  # 针对一个染色体,提取每个元素的箱子号
    #         bb=c[ii]
    #         cc=bb[1]  # cc就是数字了
    #         rr=cc[0]
    #         nn.append(rr)
    #         bb=[]
    #
    #
    #     chr.append(nn)
    #
    # print(chr)
    #
    # # print(len(boxnumlist))
    #
    #
    # boxnumlist=[]
    # for i in range(len(chr)):
    #
    #     c=max(chr[i])
    #     boxnumlist.append(c)
    #     c=[]
    # print(boxnumlist)
    #
    # print(len(boxnumlist))
    #

    # %%

    def package_calFitness(pop, max_v, boxNum):  # 爆仓检测，很重要,,      调对了！！！！！
        '''
        输入：cargo_df-货物信息,pop-个体,max_v-箱子容积,max_m-箱子在载重
        输出：适应度-fit，boxes-解码后的个体
        '''

        boxes = [[] for i in range(boxNum)]  # 这个就是solution的一个解，对应一个染色体,这里的boxnum是后面输入的boxnum[i]
        v_sum = [0] * boxNum  # 计算每一个箱子的载重
        # m_sum = [0] * boxNum

        for j in range(len(pop)):  # pop是一个染色体,改版后的

            bbb = pop[j]
            ccc = bbb[1]
            numm = ccc[0]

            box_id = int(numm) - 1  # 从0开始记，才是box_id

            dfdf = pop[j]
            dfdss = dfdf[0]
            vjjj = dfdss[0]
            v_j = vjjj  # 提取j号物品的体积,,染色体改版

            boxes[box_id].append(
                j)  # 二维数组， boxes里面的box_id位箱子（原本是[])， 装进了 j，j对应的是物品号码，这个就是我FF的solution。后面会升维，，铁定要添加物品，不需要条件判断
            v_sum[box_id] += v_j  # v_sum是一个一维数组，分别装着各个箱子的已占有空间
            # m_sum[box_id] += m_j

        vvvsum = np.array(v_sum)  # 得把v_sum转换为np矩阵    后面的fit才能计算

        num = 0  # 计数   在此之前得把v_sum转换为np矩阵    后面的fit才能计算
        for i in range(boxNum):  # 所有的箱子走一遍，检测有没有爆仓，，这个boxNum 就是箱子数目，，这个是遍历循环算法
            if (v_sum[i] <= max_v):  # max_v 就是V ，就是箱子容积
                num += 1
            else:
                break  # 跳出循环

        if num == boxNum:  # 到头了结束了，计算该解的方差
            fit = sum(vvvsum) / max_v / boxNum  # 这里直接这么用会报错，因为这种算法得把v_sum转换为np矩阵，这是必须的，切记
            # fit = 100 / (np.var(v_sum))  #
            # 构造方差和的倒数，100只是让结果数值显得大一点，懂了！！ 肯定是每一个箱子总数越接近越好，方差大说明空缺大，倒数相反
        else:
            fit = -np.var(v_sum)  # 爆仓了就无法达到num == boxNum，这个解毫无意义，那么分数变成负数，直接不入流了，垃圾解，，这应该就是唯一的判断机制了

        return round(fit, 4), boxes, v_sum  # fit 四舍五入到小数点后四位
        # boxes就是我前几天那个解，很简单的，，然后v_sum就是一个一维数组，里面是每一个箱子已经占有的空间


    def crossover(item, popsize, parent1_pops, parent2_pops, pc):

        # itemcheck=item
        child_pops = []
        for i in range(popsize):

            itemcheck = item.copy()

            parent1 = parent1_pops[i]
            parent2 = parent2_pops[i]  # 为什么parent1_pops，parent2_pops已经划分好了？两个列表了？？ 后面肯定有生成方式，，
            # 后面回来了，这两个就是分别形成的两个大小为popsize的数组

            child = []  # 初始化成 -1 ，位数同染色体位数

            if random.random() >= pc:  # 不交叉验证，就copy然后乱序
                child = parent1.copy()  # 随机生成一个
                # random.shuffle(child)     # 随意乱序，但这个和我之前的完全不一样了，是指重新委派，要不然我不打乱了？？？

            else:  # crossover发生了，，，C1策略

                thepos = random.randint(0, len(parent1) - 1)  # 这个aaaaa是脚标而不是编号，randint(0, 1)就只有0和1的

                child = parent1[0:thepos + 1]  # child[0:thepos + 1] 也是一个数组，只有child的脚标为 0,1,2 。
                # 。 thepos的元素
                # child[0:thepos + 1] 应该来自爸爸，，另外的来自妈妈

                cccc = child  # 接下来去item里面删东西,先提取不要的：child[0:thepos + 1]，也就是cccc
                needdelect1 = []
                for iq in range(len(cccc)):
                    needdelect1.append(cccc[iq][0][0])  # needdelect1应该就是child[0:thepos + 1]提取出的不要的重量，
                    # 明天早上验证，继续敲
                    # 如果没有错的话，这个needdelect1就是child节点前半部分，需要删除的部分了

                # 现在我要删除itemcheck中和cccc重叠的重量
                # 现在我要删除itemcheck中和cccc重叠的重量
                for ik in range(len(needdelect1)):
                    itemcheck.remove(needdelect1[ik])

                # 下面将要继承母亲染色体了，，如果母亲染色的该位item没有被用过，在itemcheck里面，
                # 并且item没有出现在之前的child染色体，则填充

                # 下面将要继承母亲染色体了，，如果母亲染色的该位item没有被用过，在itemcheck里面，
                # 并且item没有出现在之前的child染色体，则填充

                # 开始遍历母亲染色体：  判断还需要双层嵌套
                needdelect2 = []

                for kkk in range(len(parent2)):
                    motherweight = parent2[kkk][0][0]
                    for ikk in range(len(itemcheck)):
                        if itemcheck[ikk] == motherweight:  # itemcheck 是除掉了child的前半部分对应的物品之后的物品栏
                            # 如果在还没用的itemcheck里面找到了该物品，那么看看child前面有没有,,不用看了，因为前面是一定没有的，
                            # 因为itemcheck就是检测child的数组，itemcheck没有就一定没有

                            itemcheck[ikk] = -1  # 装好了，就把这个物品移出去 itemcheck
                            child.append(parent2[kkk])  # 可以将新的添加到child里面了
                            needdelect2.append(motherweight)

                # print(needdelect2)
                # print(child)
                # print(len(child))

                # 无敌负一法

                child_pops.append(child)

        return child_pops


    def tournament_select(pops, popsize, fit, tournament_size):
        new_pops = []  # 是不是新增染色体啊？？？
        while len(new_pops) < len(pops):
            tournament_list = random.sample(range(0, popsize),
                                            tournament_size)  # pops就是popgroup，，popsize是选取的集群的大小，然后 tournament_size就是几个里面选,等于4
            # ，tournament_list就是选出来的四个染色体对应的序号
            tournament_fit = [fit[i] for i in tournament_list]  # tournament_list每一个染色体计算fit值
            # 转化为df方便索引
            tournament_df = pd.DataFrame([tournament_list, tournament_fit]).transpose().sort_values(by=1).reset_index(
                drop=True)  # 是按照fit值排序了
            new_pop = pops[int(tournament_df.iloc[-1, 0])]  # c=tournament_df.iloc[-1, 0] ,, c就是最高分(fit)的染色体的序号
            ## 这里有一个问题，就是list问题

            new_pops.append(new_pop)

        return new_pops  # 唯一的作用，就是从pops里面抽取popsize个表现最好的染色体，，也就是new_pops



    # %%

    def package_GA(pops, max_v, item, popsize, pc, tournament_size, generations):
        # 初始化种群

        fitlist = []

        fit, boxes = [-1] * popsize, [-1] * popsize  # 每个解有一个 fit， 有一个boxes
        v_sum = [-1] * popsize  # 这个是每一个染色体的总数，popsize就是规定的种群数量，不用质疑

        boxnumlist = calculate_boxnumlist(pops)

        for h in range(len(pops)):
            boxNum = boxnumlist[h]
            pop = pops[h]
            fit[h], boxes[h], v_sum[h] = package_calFitness(pop, max_v, boxNum)

        best_fit = max(fit)  # 看回前面的fit函数
        best_pop = pops[fit.index(max(fit))].copy()  # best_fit对应的best_pop
        best_box = boxes[fit.index(max(fit))].copy()  # 每一个解是一个装箱方案，然后该装箱方案下的 best_box ，也是一个数组
        best_vsum = v_sum[fit.index(max(fit))].copy()

        if best_fit == 1: return best_pop

        iter = 0  # 迭代计数

        while iter < generations:

            pops1 = tournament_select(pops, popsize, fit, tournament_size)
            pops2 = tournament_select(pops, popsize, fit, tournament_size)

            new_pops = crossover(item, popsize, pops1, pops2, pc)
            iter += 1

            new_fit, new_boxes = [-1] * popsize, [-1] * popsize  # 初始化，记录防爆仓函数数值
            newv_sum = [-1] * popsize

            new_boxnumlist = calculate_boxnumlist(new_pops)
            for j in range(len(new_pops)):
                boxNum = new_boxnumlist[j]
                pop = new_pops[j]
                new_fit[j], new_boxes[j], newv_sum[j] = package_calFitness(pop, max_v, boxNum)

            for i in range(len(pops)):
                if fit[i] < new_fit[i]:
                    pops[i] = new_pops[i]  # 有更好的，全换，子代换掉父代
                    fit[i] = new_fit[i]
                    boxes[i] = new_boxes[i]
                    v_sum[i] = newv_sum[i]

            if best_fit < max(fit):  # 保留历史最优
                best_fit = max(fit)
                best_pop = pops[fit.index(max(fit))].copy()
                best_box = boxes[fit.index(max(fit))].copy()
                best_vsum = v_sum[fit.index(max(fit))].copy()

            fitlist.append((best_fit))

            # print("第", iter, "代适应度最优值：", best_fit)
        return best_pop, best_fit, best_box, best_vsum, fitlist


    # %%
    GAFFstart=time.time()
    if __name__ == '__main__':

        pops = pops.copy()
        max_v = 100
        item = crossitem
        popsize = 100
        pc = 0.9
        tournament_size = 4

        generations = 100

        while True:
            pop, bestfit, box, v_list, fitlist = package_GA(pops, max_v, item, popsize, pc, tournament_size, generations)
            if bestfit > 0:
                break
            else:
                boxNum += 1
    GAFFend = time.time()
    GAFFRUNTIME=GAFFend-GAFFstart

    GAFFtime.append(GAFFRUNTIME)
    GAFFmaxfit.append(bestfit)

    # %%

#%%
print(FFtime)
print(FFmaxfit)
print(GAFFtime)
print(GAFFmaxfit)


#%%
import matplotlib.pyplot as plt
sumitemlist=[]
for i in range(10,101):
    sumitemlist.append(i)



print(sumitemlist)

x1=sumitemlist
y1=FFmaxfit

x2=sumitemlist
y2=GAFFmaxfit

l1=plt.plot(x1,y1,'r--',label='FFmaxfit')
l2=plt.plot(x2,y2,'g--',label='GAFFmaxfit')

plt.plot(x1,y1,'ro-',x2,y2,'g+-')
plt.title('FFmaxfit and GAFFmaxfit')
plt.xlabel('itemnumber')
plt.ylabel('maxfit')
plt.legend()
plt.show()




#%%
import matplotlib.pyplot as plt
sumitemlist=[]
for i in range(10,101):
    sumitemlist.append(i)



print(sumitemlist)

x1=sumitemlist
y1=FFtime

x2=sumitemlist
y2=GAFFtime

l1=plt.plot(x1,y1,'r--',label='FFtime')
l2=plt.plot(x2,y2,'g--',label='GAFFtime')

plt.plot(x1,y1,'ro-',x2,y2,'g+-')
plt.title('FFtime and GAFFtime')
plt.xlabel('itemnumber')
plt.ylabel('time')
plt.legend()
plt.show()





