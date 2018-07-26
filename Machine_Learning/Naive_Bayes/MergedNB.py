from Naive_Bayes.B_Naive_Bayes_basic import *
from Naive_Bayes.MultinomialNB import MultinomialNB
from Naive_Bayes.GaussianNB import GaussianNB
from Util import DataUtil
#from Util.Timing import Timing



class MergedNB(NaiveBayes):
    """initializing structure
    self._whether_discrete:记录各个维度的变量是否是离散型变量
    self._whether_continuous:记录各个维度的变量是否是连续型变量
    self._multinomial, self._gaussian:离散型，连续型朴素贝叶斯模型
    """

    def __init__(self, whether_continuous, **kwargs):
        super().__init__(**kwargs)
        self._multinomial, self._gaussian = MultinomialNB(), GaussianNB()
        if whether_continuous is None:
            self._whether_discrete = self._whether_continuous = None
        else:
            self._whether_continuous = np.array(whether_continuous)
            self._whether_discrete = ~self._whether_continuous

     #分别利用MultinomialNB和GaussianNB的数据预处理方法进行数据预处理
    def feed_data(self, x, y, sample_weight = None):
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
        x, y, wc, features, feat_dics, label_dic = DataUtil.quantize_data(
            x, y, wc=self._whether_continuous, separate=True)
        if self._whether_continuous is None:
            self._whether_continuous = wc
            self._whether_discrete = ~self._whether_continuous
        self._label_dic = label_dic
        discrete_x, continuous_x = x
        cat_counter = np.bincount(y)
        self._cat_counter = cat_counter
        labels = [y == value for value in range(len(cat_counter))]
        #训练离散型朴素贝叶斯
        labelled_x = [discrete_x[ci].T for ci in labels]
        self._multinomial._x , self._multinomial._y = x, y
        self._multinomial._labelled_x, self._multinomial._label_zip = (
            labelled_x, list(zip(labels, labelled_x)))
        self._multinomial._cat_counter = cat_counter
        self._multinomial._feat_dics = [_dic for i, _dic in enumerate(feat_dics) if self._whether_discrete[i]]
        self._multinomial._n_possibilities = [len(feats) for i, feats in enumerate(features) if self._whether_discrete[i]]
        self._multinomial._label_dic = label_dic
        #训练连续型朴素贝叶斯
        labelled_x = [continuous_x[label].T for label in labels]
        self._gaussian._x, self._gaussian._y = continuous_x.T, y
        self._gaussian._labelled_x, self._gaussian._label_zip = labelled_x, labels
        self._gaussian._cat_counter, self._gaussian._label_dic = cat_counter, label_dic
        #处理样本权重
        self.feed_sample_weight(sample_weight)

    #分别利用MultinomialNB和GaussianNB处理样本权重的方法来处理样本权重
    def feed_sample_weight(self, sample_weight=None):
        self._multinomial.feed_sample_weight(sample_weight)
        self._gaussian.feed_sample_weight(sample_weight)

    #分别利用MultinomialNB和GaussianNB的训练函数进行训练
    def _fit(self, lb):
        self._multinomial.fit()
        self._gaussian.fit()
        p_category = self._multinomial.get_prior_probability(lb)
        discrete_func, continuous_func = self._multinomial["func"], self._gaussian["func"]

        #将MultinomialNB和GaussianNB的决策函数直接合成MergedNB决策函数
        #由于这两个决策都乘了先验概率，需要除掉一个
        def func(input_x, tar_catagory):
            input_x = np.array(input_x)
            return discrete_func(
                input_x[self._whether_discrete].astype(np.int), tar_catagory) * continuous_func(
                input_x[self._whether_continuous], tar_catagory)/p_category[tar_catagory]
        return func

    #实现转换混合型数据的方法，要注意利用MultinomialNB相应变量
    '''def _transfer_x(self, x):
        _feat_dics = self._multinomial["feat_dics"]
        idx = 0
        for d, discrete in enumerate(self._whether_discrete):
            #连续-浮点数
            if not discrete:
                x[d] = float[x[d]]
            #离散-数值化
            else:
                x[d] = _feat_dics[idx][x[d]]
            if discrete:
                idx += 1
        return x'''

    def _transfer_x(self, x):
        _feat_dics = self._multinomial["feat_dics"]
        idx = 0
        for d, discrete in enumerate(self._whether_discrete):
            if not discrete:
                x[d] = float(x[d])
            else:
                x[d] = _feat_dics[idx][x[d]]
            if discrete:
                idx += 1
        return x

if __name__ == '__main__':
    import time

    # whether_discrete = [True, False, True, True]
    # x = DataUtil.get_dataset("balloon2.0", "../../_Data/{}.txt".format("balloon2.0"))
    # y = [xx.pop() for xx in x]
    # learning_time = time.time()
    # nb = MergedNB(whether_discrete)
    # nb.fit(x, y)
    # learning_time = time.time() - learning_time
    # estimation_time = time.time()
    # nb.evaluate(x, y)
    # estimation_time = time.time() - estimation_time
    # print(
    #     "Model building  : {:12.6} s\n"
    #     "Estimation      : {:12.6} s\n"
    #     "Total           : {:12.6} s".format(
    #         learning_time, estimation_time,
    #         learning_time + estimation_time
    #     )
    # )

    whether_continuous = [False] * 16
    continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for cl in continuous_lst:
        whether_continuous[cl] = True

    train_num = 40000
    data_time = time.time()
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset(
        "bank1.0", "C:/Users/tangk/Desktop/MachineLearning-master/MachineLearning-master/_Data/bank1.0.txt", n_train=train_num)
    data_time = time.time() - data_time
    learning_time = time.time()
    nb = MergedNB(whether_continuous=whether_continuous)
    nb.fit(x_train, y_train)
    learning_time = time.time() - learning_time
    estimation_time = time.time()
    nb.evaluate(x_train, y_train)
    nb.evaluate(x_test, y_test)
    estimation_time = time.time() - estimation_time
    print(
        "Data cleaning   : {:12.6} s\n"
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            data_time, learning_time, estimation_time,
            data_time + learning_time + estimation_time
        )
    )






