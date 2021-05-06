#!/usr/bin/env python3
# coding: utf-8
# File: causality_pattern.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-12
import pyltp as ltp
import re

import utils as ut
from config import LTP_MODEL_DIR


class CompoundExtractor():
    non_comma_wp = r'[^，\s]+/[a-z]+'
    any_wp = r'[^\s]+/[a-z]+'
    comma_wp = r'，/wp'
    non_comma_cl = r'(?:{0}\s)*{0}'.format(non_comma_wp)
    any_cl = r'(?:{0}\s)*{0}'.format(any_wp)

    def __init__(self):
        # with open(LTP_MODEL_DIR+"/coord_conjs", "r", encoding="utf-8") as f:
        #     self.intra_conj = {l.strip() for l in f.readlines()}
        pass

    def rule_mid(self, sentence):
        intra_conj = {"并", "及", "同时", "也", "且", "也", "结合"}
        data = {}

        splits = re.split(r'\s{s.comma_wp}\s(?:{intra})/[cpv]\s'.format(s=self, intra="|".join(intra_conj)), sentence)
        if len(splits) > 1:
            data = {
                "tag"     : "并",
                "compound": splits,
            }
        return data

    def rule_head(self, sentence):
        head_conj = {"不仅", "不止", "不只"}
        data = {}
        pattern = re.compile(r'(?:{head}/[cpv])\s({s.non_comma_cl})\s{s.comma_wp}\s({s.any_cl})'.format(
            s=self, head="|".join(head_conj)))
        res = pattern.findall(sentence)
        if len(res) > 0:
            data = {
                "tag"     : "不仅",
                "compound": [res[0][0], res[0][1]],
            }
        return data

    def rule_pair(self, sentence):
        embed_pair = [("除", "外"), ("除", "之外"), ("除了", "外"), ("除了", "之外")]
        data = {}
        pattern = re.compile(r'({s.non_comma_cl})\s(?:{pair1}/[cpv])\s({s.non_comma_cl})\s'
                             r'(?:{pair2}/nd)?\s{s.comma_wp}\s({s.any_cl})'.format(
            s=self, pair1="|".join([p[0] for p in embed_pair]), pair2="|".join([p[1] for p in embed_pair])))
        res = pattern.findall(sentence)
        if len(res) > 0:
            data = {
                "tag"     : "除-外",
                "compound": [res[0][1], res[0][2]],
            }
        return data

    def rules_controller(self, sentence):
        infos = list()
        rules = [self.rule_mid, self.rule_head, self.rule_pair]

        for rule in rules:
            data = rule(sentence)
            if not data: continue
            infos.append(data)
            break

        return infos[0] if infos else {}

    '''返回clause坐标位置'''

    @staticmethod
    def relocate_words(sentence, clause: str):
        """
        :param clause: " ".join(word + "/" + postag)
        """
        clause_lst = [tuple(tpl.split("/")) for tpl in clause.strip().split(" ")]
        index = ut.find_sublist(sentence, clause_lst)

        return index, (index+len(clause_lst) if index != -1 else -1)

    '''抽取主控函数'''

    # use pyltp results
    def extract_pyltp(self, words, postags):
        sentence = list(zip(words, postags))
        sent = " ".join([word+"/"+postag for word, postag in sentence])
        data = self.rules_controller(sent)
        if data:
            data = {
                "tag"     : data["tag"],
                "compound": [self.relocate_words(sentence, d) for d in data["compound"]]
            }

        return data

    def has_extraction(self, words, postags):
        sentence = list(zip(words, postags))

        sent = " ".join([word+"/"+postag for word, postag in sentence])
        datas = self.rules_controller(sent)

        return bool(len(datas))


default_compound_extractor = CompoundExtractor()
'''测试'''


def run():
    segmentor = ltp.Segmentor()
    segmentor.load_with_lexicon(LTP_MODEL_DIR+"/cws.model", LTP_MODEL_DIR+"/lexicon")

    postagger = ltp.Postagger()
    postagger.load(LTP_MODEL_DIR+"/pos.model")

    parser = ltp.Parser()
    parser.load(LTP_MODEL_DIR+"/parser.model")

    extractor = CompoundExtractor()

    sent = [
        "错峰生产及市场库存水位偏低，",
    ]

    for s in sent:
        words = segmentor.segment(s)
        postags = postagger.postag(words)
        data = extractor.extract_pyltp(words, postags)
        left, right = data["left"], data["right"]
        print("***********************")
        print("left:", "".join(words[left[0]:left[1]]))
        print("right:", "".join(words[right[0]:right[1]]))


if __name__ == '__main__':
    run()
