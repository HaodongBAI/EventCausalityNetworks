# coding=utf-8
from itertools import product
from collections import defaultdict
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


def get_Chinese_font():
    return FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')


def combination_dict(dict):
    lsts = [[(k, vv) for vv in v] for k, v in dict.items()]
    for tupled_setting in product(*lsts):
        yield {k: v for (k, v) in tupled_setting}
    return


def distinct(lst):
    return list(set(lst))


def flatten(x): return [y for l in x for y in flatten(l)] if isinstance(x, list) else [x]


def map_list(func, x): return list(map(func, x))


def filter_list(func, x): return list(filter(func, x))


def filter_dict(func, x): return {k: v for k, v in x.items() if func(k, v)}


def order_by_func():
    return


def where(x, func):
    return filter_list(lambda i: func(i, x[i]), range(len(x)))


def find(x, func):
    for i, ele in enumerate(x):
        if func(i, ele):
            return i
    else:
        return None


def group_by_key(x, key, value):
    if len(x) == 0: return dict()
    res = defaultdict(list)  # 不可以用set,不一定可以hash
    for t in x:
        res[key(t)].append(value(t))
    return res


def transpose_dict(dual_dict):
    res = {}
    for k, d in dual_dict.items():
        for c, v in d.items():
            if c not in res:
                res[c] = {}
            res[c][k] = v
    return res


def reduce_by_key():
    return


def reduce_value():
    return


def map_value():
    return


def build_mapping_from_list(lst):
    ele2id = {}
    id2ele = {}
    for ele in lst:
        if ele not in ele2id:
            ele2id[ele] = len(ele2id)
            id2ele[ele2id[ele]] = ele

    return ele2id, id2ele


def dict_union(dct1, dct2):
    res = deepcopy(dct2)
    res.update(dct1)
    return res


def split_list(lst, func):
    index = where(lst, func)
    if index:
        s_index = [0] + map_list(lambda i: i + 1, index)
        e_index = map_list(lambda i: i, index) + [len(lst)]
        if index[0] == 0:
            s_index, e_index = s_index[1:], e_index[1:]
        if index[-1] == len(lst) - 1:
            s_index, e_index = s_index[:-1], e_index[:-1]
        return s_index, e_index
    else:
        return [0, ], [len(lst), ]


def find_sublist(lst, sub_lst, eq=lambda i, j: i == j):
    def generate_pnext(sub_lst):
        index, m = 0, len(sub_lst)
        pnext = [0]*m
        i = 1
        while i < m:
            if eq(sub_lst[i], sub_lst[index]):
                pnext[i] = index + 1
                index += 1
                i += 1
            elif index != 0:
                index = pnext[index - 1]
            else:
                pnext[i] = 0
                i += 1
        return pnext

    pnext = generate_pnext(sub_lst)
    n = len(lst)
    m = len(sub_lst)
    i, j = 0, 0
    while (i < n) and (j < m):
        if eq(lst[i], sub_lst[j]):
            i += 1
            j += 1
        elif j != 0:
            j = pnext[j - 1]
        else:
            i += 1
    if j == m:
        return i - j
    else:
        return -1


def window(lst, w):
    for e in range(w, len(lst) + 1):
        yield lst[e - w:e]


class UnionFind(object):
    """并查集类"""

    def __init__(self, lst):
        """长度为n的并查集"""
        n = len(lst)
        self.ele2id, self.id2ele = build_mapping_from_list(lst)
        self.uf = [-1 for i in range(n)]
        self.sets_count = n

    def find_index(self, pi):
        if self.uf[pi] < 0:
            return pi
        self.uf[pi] = self.find_index(self.uf[pi])
        return self.uf[pi]

    def find(self, p):
        """尾递归"""
        p = self.ele2id[p]
        if self.uf[p] < 0:
            return p
        self.uf[p] = self.find_index(self.uf[p])
        return self.uf[p]

    def union(self, p, q):
        """连通p,q 让q指向p"""
        proot = self.find(p)
        qroot = self.find(q)
        if proot == qroot:
            return
        elif self.uf[proot] > self.uf[qroot]:  # 负数比较, 左边规模更小
            self.uf[qroot] += self.uf[proot]
            self.uf[proot] = qroot
        else:
            self.uf[proot] += self.uf[qroot]  # 规模相加
            self.uf[qroot] = proot
        self.sets_count -= 1  # 连通后集合总数减一

    def self_cluster(self, p):
        p = self.ele2id[p]
        if self.uf[p] == -1:
            return True
        return False

    def is_connected(self, p, q):
        """判断pq是否已经连通"""
        return self.find(p) == self.find(q)  # 即判断两个结点是否是属于同一个祖先

    @property
    def root2ele(self):
        return group_by_key(list(enumerate(self.uf)),
                            key=lambda tpl: self.id2ele[tpl[0] if tpl[1] < 0 else self.find_index(tpl[0])],
                            value=lambda tpl: self.id2ele[tpl[0]])

    @property
    def ele2root(self):
        return transpose_dict(self.root2ele)


if __name__ == '__main__':
    print(split_list([1, 2, 3, 4, 5, 7], lambda i, t: t > 10))
