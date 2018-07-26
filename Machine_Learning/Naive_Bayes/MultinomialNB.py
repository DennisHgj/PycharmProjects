import PythonProject.PycharmProjects.Machine_Learning.Naive_Bayes.B_Naive_Bayes_basic as NBb
import numpy as np
import time
from PythonProject.PycharmProjects.Machine_Learning.Util import DataUtil
import matplotlib.pyplot as plt

import pylab as mpl


class MultinomialNB(NBb.NaiveBayes):
    """定义预处理数据的方法"""

    def feed_data(self, x, y, sample_weight=None):
        """分情况将输入向量x进行转置"""
        """
        if isinstance(x, list):
            features = map(list, zip(*x))
            #unzip x [(1,2,3),(4,5,6)]=>(1,2,3)(4,5,6)
        else:
            features = x.T"""
        """利用python中内置的高级函数结构--集合，获取各个维度的特征和类别种类
        为了利用bincount方法来优化算法，将所有特征从0开始数值化
        注意：需要将数值化过程中的转换关系记录成字典，否则无法对新数据进行判断
        """
        """features = [set(feat) for feat in features]
        feat_dics = [{_l: i for i, _l in enumerate(feats)} for feats in features]
        label_dic = {_l: i for i, _l in enumerate(set(y))}
        # 利用转换字典更新数据集
        x = np.array([[feat_dics[i][_l] for i, _l in enumerate(sample)] for sample in x])
        y = np.array([label_dic[yy] for yy in y])"""
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        x, y, _, features, feat_dics, label_dic = DataUtil.quantize_data(x, y, wc=np.array([False] * len(x[0])))
        # 利用Numpy的bincount方法，获取各类别数据的个数
        cat_counter = np.bincount(y)
        # 记录各个维度特征的取值个数
        n_possibilities = [len(feats) for feats in features]
        # 获得各类别数据的下标
        labels = [y == value for value in range(len(cat_counter))]
        # 利用下标获取记录按类别分开后的输入数据的数组
        labelled_x = [x[ci].T for ci in labels]
        # 更新模型的各个属性
        self._x, self._y = x, y
        self._labelled_x, self._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._cat_counter, self._feat_dics, self._n_possibilities = cat_counter, feat_dics, n_possibilities
        self._label_dic = {i: _l for _l, i in label_dic.items()}
        # 调用处理样本的权重函数，以更新记录条件概率的数组
        self.feed_sample_weight(sample_weight)

    # 定义处理样本权重的函数/
    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = []
        # 利用Numpy的bincount方法获取带权重的条件概率极大似然估计
        for dim, _p in enumerate(self._n_possibilities):
            if sample_weight is None:
                self._con_counter.append([np.bincount(xx[dim], minlength=_p) for xx in self._labelled_x])
            else:
                self._con_counter.append([
                    np.bincount(xx[dim], weights=sample_weight[label] / sample_weight[label].mean(), minlength=_p)
                    for label, xx in self._label_zip])

    # 定义核心训练函数
    def _fit(self, lb):
        n_dim = len(self._n_possibilities)
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)

        # data即为存储加了平滑项后的条件概率的数组
        data = [None] * n_dim

        for dim, n_possibilities in enumerate(self._n_possibilities):
            data[dim] = [[
                (self._con_counter[dim][c][p] + lb) / (self._cat_counter[c] + lb * n_possibilities)
                for p in range(n_possibilities)
            ] for c in range(n_category)]
        self._data = [np.asarray(dim_info) for dim_info in data]

        # 利用data生成决策函数
        '''def func(input_x, tar_category):
            rs = 1
            # 遍历各个维度，利用data 和条件独立性假设计算联合条件概率
            for d, xx in enumerate(input_x):
                rs *= data[d][tar_category][xx]
            return rs * p_category[tar_category]

        # 返回决策函数
        return func'''

        '''向量化MultinomialNB中的决策函数'''

        def func(input_x, tar_category):
            '''将输入换成二维数组（矩阵）'''
            input_x = np.atleast_2d(input_x).T
            """使用向量存储结果，初始化为全1向量"""
            rs = np.ones(input_x.shape[1])
            for d, xx in enumerate(input_x):
                """虽然代码没变，但此处_x是一个向量而不是数
                因使用Numpy 故此语法合理且高效"""
                rs *= data[d][tar_category][xx]
            return rs * p_category[tar_category]
        return func

    # 定义数值化数据的函数
    def _transfer_x(self, x):
        """遍历每个元素，利用转换字典进行数值化"""
        for j, char in enumerate(x):
            x[j] = self._feat_dics[j][char]
        return x

    def visualize(self, save=False):
        colors = plt.cm.Paired([i / len(self._label_dic) for i in range(len(self._label_dic))])
        colors = {cat: color for cat, color in zip(self._label_dic.values(), colors)}
        rev_feat_dicts = [{val: key for key, val in feat_dict.items()} for feat_dict in self._feat_dics]
        for j in range(len(self._n_possibilities)):
            rev_dict = rev_feat_dicts[j]
            sj = self._n_possibilities[j]
            tmp_x = np.arange(1, sj + 1)
            title = "$j = {}; S_j = {}$".format(j + 1, sj)
            plt.figure()
            plt.title(title)
            for c in range(len(self._label_dic)):
                plt.bar(tmp_x - 0.35 * c, self._data[j][c, :], width=0.35,
                        facecolor=colors[self._label_dic[c]], edgecolor="white",
                        label=u"class: {}".format(self._label_dic[c]))
            plt.xticks([i for i in range(sj + 2)], [""] + [rev_dict[i] for i in range(sj)] + [""])
            plt.ylim(0, 1.0)
            plt.legend()
            if not save:
                plt.show()
            else:
                plt.savefig("d{}".format(j + 1))


if __name__ == '__main__':
    for dataset in ("balloon1.0", "balloon1.5"):
        """读入数据"""
        # a = DataUtil()
        _x, _y = DataUtil.get_dataset(dataset, "C:\\Users\\tangk\\PycharmProjects\Machine_Learning\\_Data\\{}.txt".
                                      format(dataset))

        # 实例化模型并进行训练、同时记录整个过程花费的时间
        learning_time = time.time()
        nb = MultinomialNB()
        nb.fit(_x, _y)
        # print(nb.fit(_x, _y))
        learning_time = time.time() - learning_time
        # 评估模型的表现，同时记录评估评估过程花费的时间
        print("=" * 30)
        print(dataset)
        print("-" * 30)
        estimation_time = time.time()
        nb.evaluate(_x, _y)
        estimation_time = time.time() - estimation_time
        # 将记录下来的耗时输出
        print(
            "Model building : {:12.6} s\n"
            "Estimation :     {:12.6} s\n"
            "Total :          {:12.6} s".format(
                learning_time, estimation_time,
                learning_time + estimation_time
            )
        )
    print("=" * 30)
    print("mushroom")
    print("-" * 30)

    train_num = 6000
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset(
        "mushroom", "C:\\Users\\tangk\\PycharmProjects\Machine_Learning\\_Data\\mushroom.txt", n_train=train_num,
        tar_idx=0)
    learning_time = time.time()
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    learning_time = time.time() - learning_time

    estimation_time = time.time()
    nb.evaluate(x_train, y_train)
    nb.evaluate(x_test, y_test)
    estimation_time = time.time() - estimation_time

    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            learning_time, estimation_time,
            learning_time + estimation_time
        )
    )
    # nb.show_timing_log()
    nb.visualize()

'''mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes,unicode_minus'] = False
data = nb["data"]
colors = {"e": "lightSkyBlue", "p": "orange"}
_rev_feat_dics = [{_val: _key for _key, _val in _feat_dic.items()} for _feat_dic in self._feat_dics]'''
