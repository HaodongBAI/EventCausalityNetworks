# coding: utf-8
import pyltp as ltp
import re
import functools
import utils as ut
from config import LTP_MODEL_DIR


class CausalityExtractor():
    non_comma_wp = r'[^，\s]+/[a-z]+'
    any_wp = r'[^\s]+/[a-z]+'
    comma_wp = r'，/wp'
    non_comma_cl = r'(?:{0}\s)*{0}'.format(non_comma_wp)
    any_cl = r'(?:{0}\s)*{0}'.format(any_wp)

    def __init__(self):
        pass

    '''由因到果配套式'''

    def rule_pair(self, sentence):
        '''
        conm1:〈因为,从而〉、〈因为,为此〉、〈既[然],所以〉、〈因为,为此〉、〈由于,为此〉、〈只有|除非,才〉、〈由于,以至[于]>、〈既[然],却>、
        〈如果,那么|则〉、<由于,从而〉、<既[然],就〉、〈既[然],因此〉、〈如果,就〉、〈只要,就〉〈因为,所以〉、 <由于,于是〉、〈因为,因此〉、
         <由于,故〉、 〈因为,以致[于]〉、〈因为,因而〉、〈由于,因此〉、<因为,于是〉、〈由于,致使〉、〈因为,致使〉、〈由于,以致[于] >
         〈因为,故〉、〈因[为],以至[于]>,〈由于,所以〉、〈因为,故而〉、〈由于,因而〉
        conm1_model:<Conj>{Cause}, <Conj>{Effect}
        '''
        datas = list()
        word_pairs = [['因为', '从而'], ['因为', '为此'], ['既然?', '所以'],
                      ['因为', '为此'], ['由于', '为此'], ['除非', '才'],
                      ['只有', '才'], ['由于', '以至于?'], ['既然?', '却'],
                      ['如果', '那么'], ['如果', '则'], ['由于', '从而'],
                      ['既然?', '就'], ['既然?', '因此'], ['如果', '就'],
                      ['只要', '就'], ['因为', '所以'], ['由于', '于是'],
                      ['因为', '因此'], ['由于', '故'], ['因为', '以致于?'],
                      ['因为', '以致'], ['因为', '因而'], ['由于', '因此'],
                      ['因为', '于是'], ['由于', '致使'], ['因为', '致使'],
                      ['由于', '以致'], ['因为', '故'], ['因为?', '以至'],
                      ['由于', '所以'], ['因为', '故而'], ['由于', '因而']]
        for word in word_pairs:
            pattern = re.compile(r'\s?({word[0]})/[p|c]\s({s.any_cl})\s({word[1]})/[p|c]\s({s.any_cl})'.format(
                word=word, s=self))
            result = pattern.findall(sentence)
            for res in result:
                datas.append({
                    "tag"   : res[0] + '-' + res[2],
                    "cause" : [res[1], ],
                    "effect": [res[3], ]
                })

        patterns = [

            (re.compile(
                r'({s.non_comma_cl})\s在/p\s({s.non_comma_cl})\s(?:之/u\s)?[下后时]/nd?(?:\s{s.comma_wp})?\s'
                r'({s.any_cl})'.format(
                    s=self)), [1], [0, 2]),
            (re.compile(r'({s.non_comma_cl})\s(?:之/u\s)?[下]/nd?\s{s.comma_wp}\s({s.any_cl})'.format(
                s=self)), [0], [1]),
        ]
        for pattern, cause_index, effect_index in patterns:
            result = pattern.findall(sentence)
            if len(result) == 0: continue
            for res in result:
                datas.append({
                    "tag"   : "在-下",
                    "cause" : [res[c] for c in cause_index],
                    "effect": [res[e] for e in effect_index]
                })
            break
        return self.longest_triplet(datas, sentence)

    '''由因到果居中式明确'''

    def rule_mid(self, sentence):
        '''
        cons2:于是、所以、故、致使、以致[于]、因此、以至[于]、从而、因而
        cons2_model:{Cause},<Conj...>{Effect}
        '''

        verbs = ["有助", "导致", "使", "促成", "造成", "引导", "促使", "酿成", "引发", "促进", "引起", "引来", "引致", "诱发", "推动",
                 "影响", "致使", "使得", "带来", "波及", "诱使", "将使", "带动"]

        conjs = ['于是', '所以', '故', '致使', '以致', '因此', '以至', '从而', '因而', '以免', '以便', '为此', '才',
                 '以免', '以便', '为此', '才']

        patterns = [
            (re.compile(
                r'({s.any_cl})\s{s.comma_wp}\s(?:{s.non_comma_wp}\s)?({cs})/[pcv]\s({s.any_cl})'.format(
                    s=self, cs="|".join(conjs + verbs)
                )), [0], [1]),
        ]

        datas = list()
        for pattern, ci, ei in patterns:
            result = pattern.findall(sentence)
            for res in result:
                datas.append({
                    "tag"   : res[1],
                    "cause" : [res[i] for i in ci],
                    "effect": [res[i] for i in ei]
                })
        return self.longest_triplet(datas, sentence)

    '''由因到果前端式模糊'''

    def rule_head(self, sentence):
        '''
        prep:为了、依据、为、按照、因[为]、按、依赖、照、比、凭借、由于
        prep_model:<Prep...>{Cause},{Effect}
        '''
        conjs = ['为了', '按照', '因为', '因', '按', '依赖', '由于', '随着', '如果', '只要', "经过", "对于", "面对"]

        pattern = re.compile(r'({cs})/[p|c]\s({s.non_comma_cl})\s{s.comma_wp}\s({s.any_cl})'.format(
            s=self, cs="|".join(conjs)
        ))
        result = pattern.findall(sentence)
        datas = list()
        for res in result:
            datas.append({
                "tag"   : res[0],
                "cause" : [res[1]],
                "effect": [res[2]]
            })

        patterns = [
            (re.compile(r"^受[惠到]/[vp](?:\s于/p)?\s({s.non_comma_cl})\s{s.comma_wp}\s({s.any_cl})".format(s=self)),
             [0], [1]),
            (re.compile(
                r"({s.non_comma_cl})\s受[惠到]/[vp](?:\s于/p)?\s({s.non_comma_cl})\s{s.comma_wp}\s({s.any_cl})".format(
                    s=self)), [1], [0, 2]),
        ]
        for pattern, cause_index, effect_index in patterns:
            result = pattern.findall(sentence)
            if len(result) == 0: continue
            for res in result:
                datas.append({
                    "tag"   : "受惠/受到",
                    "cause" : [res[c] for c in cause_index],
                    "effect": [res[e] for e in effect_index]
                })
            break
        return self.longest_triplet(datas, sentence)

    '''返回clause坐标位置'''

    @staticmethod
    def slash_split(s):
        slash_index = ut.where(s, lambda _, c: c == "/")
        if len(slash_index) == 0:
            raise ValueError("There is no slash in (word, postag) pair [{}].".format(s))

        return s[:slash_index[-1]], s[slash_index[-1] + 1:]

    @staticmethod
    def relocate_words(sentence, clause: str):
        """
        :param clause: " ".join(word + "/" + postag)
        """
        clause_lst = [CausalityExtractor.slash_split(tpl) for tpl in clause.strip().split(" ")]
        index = ut.find_sublist(sentence, clause_lst)
        if index == -1:
            raise ValueError("Sub-list not found: {}. Original list is {}.".format(clause, sentence))
        return index, index + len(clause_lst)

    @staticmethod
    def longest_triplet(datas, sent):
        if len(datas) > 1:
            if len(set([data["tag"] for data in datas])) > 1:
                ValueError("Too many tags for one segments: {}\n\tTags: {}".format(
                    sent, ",".join([data["tag"] for data in datas])))

            datas_aug = [{"cause": " ".join(data["cause"]), "effect": " ".join(data["effect"])} for data in datas]
            cmp = lambda s, t: len(s["cause"]) <= len(t["cause"]) and len(s["effect"]) <= len(t["effect"])
            data = datas[functools.reduce(lambda i, j: j if cmp(datas_aug[i], datas_aug[j]) else i, range(len(datas)))]
        elif len(datas) == 1:
            data = datas[0]
        else:
            data = {}
        return data

    '''抽取控制函数'''

    def rules_controller(self, sentence):
        infos = list()
        rules = [self.rule_pair,
                 # self.rule_mid,
                 self.rule_head]

        for rule in rules:
            data = rule(sentence)
            if not data: continue
            infos.append(data)
            break

        return infos

    '''抽取主控函数'''

    # use pyltp results
    def extract_pyltp(self, words, postags):
        sentence = list(zip(words, postags))

        sent = " ".join([word + "/" + postag for word, postag in sentence])
        datas = self.rules_controller(sent)

        data = self.longest_triplet(datas, sentence)
        if data:
            data = {
                "tag"   : data["tag"],
                "cause" : [self.relocate_words(sentence, cause) for cause in data["cause"]],
                "effect": [self.relocate_words(sentence, effect) for effect in data["effect"]]
            }
        return data

    def has_extraction(self, words, postags):
        sentence = list(zip(words, postags))

        sent = " ".join([word + "/" + postag for word, postag in sentence])
        datas = self.rules_controller(sent)

        return bool(len(datas))


default_causality_extractor = CausalityExtractor()
'''测试'''


def run():
    segmentor = ltp.Segmentor()
    segmentor.load_with_lexicon(LTP_MODEL_DIR + "/cws.model", LTP_MODEL_DIR + "/lexicon")

    postagger = ltp.Postagger()
    postagger.load(LTP_MODEL_DIR + "/pos.model")

    parser = ltp.Parser()
    parser.load(LTP_MODEL_DIR + "/parser.model")

    extractor = CausalityExtractor()

    sent = [
        # "受惠成品鞋出货量成长，丰泰(9910)1月营收53.05亿元，年增21.17%，带动获利年增逾六成，单月EPS0.49元",
        # "台泥(1101)受惠中国“错峰生产、减量保价”政策，前三季水泥价平量涨，每公吨平均涨价幅度约33%，第三季毛利率为27%，EPS为1.2元，累计前三季获利更大增2.22倍，每股盈余3.35元，再度创下过去10"
        # "年的最好表现",
        "首季获利大增主要受惠错峰生产到位，售价提高79人民币，水泥吨毛利大幅提升",
    ]

    for s in sent:
        words = segmentor.segment(s)
        postags = postagger.postag(words)
        data = extractor.extract_pyltp(words, postags)
        print("***********************")
        print("cause:", "".join(["".join(words[c[0]:c[1]]) for c in data["cause"]]))
        print("effect:", "".join(["".join(words[e[0]:e[1]]) for e in data["effect"]]))


if __name__ == '__main__':
    run()
