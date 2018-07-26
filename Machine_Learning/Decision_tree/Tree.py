from copy import deepcopy
from Decision_tree.Node import *
import numpy as np


class CvDBase:
    """
    初始化
    self.nodes：记录所有Node的列表
    self.roots:主要用于CART剪枝的属性（用于存储算法过程中产生的各个决策树）
    self.max_depth： 记录决策树最大深度的属性
    self.root，self.feature_sets:根节点和记录可选特征维度的列表
    self.label_dic:和朴素贝叶斯里面相应的属性意义一致、是类别的转换字典
    self.prune,self.layers：主要用于ID3和C4.5剪枝的两个属性（1，惩罚因子，2记录每一层的Node）
    self.whether_continuous:记录着各个维度的特征是否连续的列表
    """

    def __init__(self, whether_continuous=None, max_depth=None, node=None):
        self.nodes, self.layers, self.roots = [], [], []
        self.max_depth = max_depth
        self.root = node
        self.feature_sets = []
        self.label_dic = {}
        self.prune_alpha = 1
        self.whether_continuous = whether_continuous

    def __str__(self):
        return "CvDTree({})".format(self.root.height)

    __repr__ = __str__

    def feed_data(self, x, continuous_rate=0.2):
        """利用set获取各个维度特征的所有可能取值"""
        self.feature_sets = [set(dimension) for dimension in x.T]
        data_len, data_dim = x.shape
        """判断是否连续"""
        self.whether_continuous = np.array([len(feat) >= continuous_rate * data_len for feat in self.feature_sets])
        self.root.feats = [i for i in range(x.shape[1])]
        self.root.feed_tree(self)
        """feed_tree方法
        让决策树中所有的Node记录他们所属的Tree结构
        将自己记录在tree中记录所有的Node的列表nodes里
        根据tree的相应属性更新记录连续特征的列表"""

    """参数alpha与剪枝有关，可按下不表
    cv_rate用于控制交叉验证集的大小，train_only则控制程序是否进行数据集的切分"""

    def fit(self, x, y, alpha=None, sample_weight=None, eps=1e-8, cv_rate=0.2, train_only=False):
        """数值化类别向量"""
        _dic = {c: i for i, c in enumerate(set(y))}
        y = np.array([_dic[yy] for yy in y])
        self.label_dic = {value: key for key, value in _dic.items()}
        x = np.array(x)
        """根据特征个数定出alpha"""
        self.prune_alpha = alpha if alpha is not None else x.shape[1] / 2
        """如果需要划分数据集的话"""
        if not train_only and self.root.is_cart:
            """根据cv_rate将数据集随机分成训练集和交叉验证集
            实现的核心思想是利用下标来切分"""
            _train_num = int(len(x) * (1 - cv_rate))
            _indices = np.random.permutation(np.arange(len(x)))
            _train_indices = _indices[:_train_num]
            _test_indices = _indices[_train_num:]
            if sample_weight is not None:
                """对切分后的样本权重做归一化处理"""
                _train_weights = sample_weight[_train_indices]
                _test_weights = sample_weight[_test_indices]
                _train_weights /= np.sum(_train_weights)
                _test_weights /= np.sum(_test_weights)
            else:
                _train_weights = _test_weights = None
            x_train, y_train = x[_train_indices], y[_train_indices]
            x_cv, y_cv = x[_test_indices], y[_test_indices]
        else:
            x_train, y_train, _train_weights = x, y, sample_weight
            x_cv = y_cv = _test_weights = None
        self.feed_data(x_train)
        """调用根节点生成算法"""
        self.root.fit(x_train, y_train, _train_weights, eps)
        """调用对Node剪枝算法的封装"""
        self.prune(x_cv, y_cv, _test_weights)

    def reduce_nodes(self):
        for i in range(len(self.nodes) - 1, -1, -1):
            if self.nodes[i].pruned:
                self.nodes.pop(i)

    def _update_layers(self):
        """根据整颗决策树的高度，在self.layers里面放相应的列表"""
        self.layers = [[] for _ in range(self.root.height)]

    def _prune(self):
        self._update_layers()
        _tmp_nodes = []
        """更新完决策树每一层的Node后，从后往前的向_tmp_nodes中加Node"""
        for _node_lst in self.layers[::-1]:
            for _node in _node_lst[::-1]:
                if _node.category is None:
                    _tmp_nodes.append(_node)
        _old = np.array([node.cost() + self.prune_alpha * len(node.leafs) for node in _tmp_nodes])
        _new = np.array([node.cost(pruned=True) + self.prune_alpha for node in _tmp_nodes])
        """使用_mask变量存储_old和_new对应位置的大小关系"""
        _mask = _old >= _new
        while True:
            """若只剩根节点就退出循环体"""
            if self.root.height == 1:
                return
            p = np.argmax(_mask)
            """如果_new中有比_old中对应损失小的损失、则进行局部剪枝"""
            if _mask[p]:
                _tmp_nodes[p].prune()
                """根据被影响了的Node，更新_old,_mask对应位置的值"""
                for i, node in enumerate(_tmp_nodes):
                    if node.affected:
                        _old[i] = node.cost() + self.prune_alpha * len(node.leafs)
                        _mask[i] = _old[i] >= _new[i]
                        node.affected = False
                """根据被剪掉的Node，将各个变量对应的位置除去（从前往后）"""
                for i in range(len(_tmp_nodes) - 1, -1, -1):
                    if _tmp_nodes[i].pruned:
                        _tmp_nodes.pop(i)
                        _old = np.delete(_old, i)
                        _new = np.delete(_new, i)
                        _mask = np.delete(_mask, i)
            else:
                break
        self.reduce_nodes()

    def _cart_prune(self):
        """暂时将所有的节点记录所属Tree的属性置为None"""
        self.root.cut_tree()
        _tmp_nodes = [node for node in self.nodes if node.category is None]
        _thresholds = np.array([node.get_threshold() for node in _tmp_nodes])
        while True:
            """利用deepcopy对当前的根节点进行深拷贝，存入self.roots列表
            如果前面没有把记录Tree的属性设为None，这里也会对树做深拷贝，会引发严重的内存问题，速度也会被拖慢非常多"""
            root_copy = deepcopy(self.root)
            self.roots.append(root_copy)
            if self.root.height == 1:
                break
            p = np.argmin(_thresholds)
            _tmp_nodes[p].prune()
            for i, node in enumerate(_tmp_nodes):
                """更新被影响的Node的阈值"""
                if node.affected:
                    _thresholds[i] = node.get_threshold()
                    node.affected = False
                for i in range(len(_tmp_nodes) - 1, -1, -1):
                    """去掉列表相应元素"""
                    if _tmp_nodes[i].pruned:
                        _tmp_nodes.pop(i)
                        _thresholds = np.delete(_thresholds, i)
        self.reduce_nodes()

    @staticmethod
    def acc(y, y_pred, weights):
        """使用加权正确率作为交叉验证的方法"""
        if weights is not None:
            return np.sum((np.array(y) == np.array(y_pred)) * weights) / len(y)
        return np.sum(np.array(y) == np.array(y_pred)) / len(y)

    def prune(self, x_cv, y_cv, weights):
        if self.root.is_cart:
            """验证是否传入交叉验证集"""
            if x_cv is not None and y_cv is not None:
                self._cart_prune()
                _arg = np.argmax([CvDBase.acc(y_cv, tree.predict(x_cv), weights) for tree in self.roots])
                _tar_root = self.roots[_arg]
                """由于Node的feed_tree方法会递归地更新nodes属性，所以要先重置"""
                self.nodes = []
                _tar_root.feed_tree(self)
                self.root = _tar_root
            else:
                self._prune()


class CvDMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        _, _node = bases

        def __init__(self, whether_continuous=None, max_depth=None, node=None, **_kwargs):
            tmp_node = node if isinstance(node, CvDNode) else _node
            CvDBase.__init__(self, whether_continuous, max_depth, tmp_node(**_kwargs))
            self._name = name

        attr["__init__"] = __init__
        return type(name, bases, attr)


class ID3Tree(CvDBase, ID3Node, metaclass=CvDMeta):
    pass


class C45Tree(CvDBase, C45Node, metaclass=CvDMeta):
    pass


class CartTree(CvDBase, CartNode, metaclass=CvDMeta):
    pass
