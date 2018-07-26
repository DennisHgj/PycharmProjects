from Naive_Bayes.B_Naive_Bayes_basic import *
from Naive_Bayes.NBFunctions import *
import matplotlib.pyplot as plt
from Util import DataUtil
from Naive_Bayes.MultinomialNB import MultinomialNB


class GaussianNB(NaiveBayes):
    def feed_data(self, x, y, sample_weight=None):
        """简单的调用Python自带的float方法将输入数据数值化"""
        x = np.array([list(map(lambda c: float(c), sample))for sample in x])
        """数值化类别向量"""
        labels = list(set(y))
        label_dic = {label: i for i, label in enumerate(labels)}
        y = np.array([label_dic[yy] for yy in y])
        cat_counter = np.bincount(y)
        labels = [y == value for value in range(len(cat_counter))]
        labelled_x = [x[label].T for label in labels]
        #更新模型各个属性
        self._x, self._y = x.T, y
        self._labelled_x,  self._label_zip = labelled_x, labels
        self._cat_counter, self._label_dic = cat_counter, {i: _l for _l, i in label_dic.items()}
        self.feed_sample_weight(sample_weight)

    #定义处理样本权重的函数
    def feed_sample_weight(self, sample_weight=None):
        if sample_weight is not None:
            local_weights = sample_weight * len(sample_weight)
            for i, label in enumerate(self._label_zip):
                self._labelled_x[i] *= local_weights[label]

    def _fit(self, lb):
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)
        #利用极大似然估计获得计算条件概率的函数，使用数组变量data进行存储
        data = [NBFunctions.gaussian_maximum_likelihood(
            self._labelled_x, n_category, dim) for dim in range(len(self._x))]
        self._data = data

        def func(input_x, tar_category):
            rs = 1
            for d, xx in enumerate(input_x):
                """由于data中存储的是函数，所以需要调用它来进行条件概率计算"""
                rs *= data[d][tar_category](xx)
            return rs * p_category[tar_category]
        return func


    def visualize(self, save=False):
        colors = plt.cm.Paired([i / len(self._label_dic) for i in range(len(self._label_dic))])
        colors = {cat: color for cat, color in zip(self._label_d.values(), colors)}
        for j in range(len(self._x)):
            tmp_data = self._x[j]
            x_min, x_max = np.min(tmp_data), np.max(tmp_data)
            gap = x_max - x_min
            tmp_x = np.linspace(x_min-0.1*gap, x_max+0.1*gap, 200)
            title = "$j = {}$".format(j + 1)
            plt.figure()
            plt.title(title)
            for c in range(len(self._label_dic)):
                plt.plot(tmp_x, [self._data[j][c](xx) for xx in tmp_x],
                         c=colors[self._label_dic[c]], label="class: {}".format(self._label_dic[c]))
            plt.xlim(x_min-0.2*gap, x_max+0.2*gap)
            plt.legend()
            if not save:
                plt.show()
            else:
                plt.savefig("d{}".format(j + 1))

    @staticmethod
    def _transfer_x(x):
        return x
if __name__ == '__main__':
    import time

    xs, ys = DataUtil.get_dataset("mushroom","C:\\Users\\tangk\\PycharmProjects\Machine_Learning\\_Data\\mushroom.txt", tar_idx=0)
    nb = MultinomialNB()
    nb.feed_data(xs, ys)
    xs, ys = nb["x"].tolist(), nb["y"].tolist()

    train_num = 6000
    x_train, x_test = xs[:train_num], xs[train_num:]
    y_train, y_test = ys[:train_num], ys[train_num:]

    learning_time = time.time()
    gb = GaussianNB()
    gb.fit(x_train, y_train)
    learning_time = time.time() - learning_time

    estimation_time = time.time()
    gb.evaluate(x_train, y_train)
    gb.evaluate(x_test, y_test)
    estimation_time = time.time() - estimation_time

    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            learning_time, estimation_time,
            learning_time + estimation_time
        )
    )
    gb.visualize()