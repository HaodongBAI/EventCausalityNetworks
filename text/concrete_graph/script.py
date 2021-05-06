import pyltp as ltp
import re
from itertools import product

import networkx as nx

import utils as ut
from concrete_graph.clause_similarity import clause_similarity, fuse_clause_graphs, node_merger_wrapper, stop_list
from concrete_graph.paratactic_sentence_graph import build_sentence_graph, plot_graph
from config import LTP_MODEL_DIR
from database_model.loader import load_efficient_metas, load_story_from_announcement_id, load_story_from_metas
from preprocess import line_return_preprocess
import os
from config import CONCRETE_GRAPH_DIR
import json
from collections import defaultdict


def test_sent(sent):
    segmentor = ltp.Segmentor()
    segmentor.load_with_lexicon(LTP_MODEL_DIR + "/cws.model", LTP_MODEL_DIR + "/lexicon")

    postagger = ltp.Postagger()
    postagger.load_with_lexicon(LTP_MODEL_DIR + "/pos.model", LTP_MODEL_DIR + "/postag_lexicon")

    parser = ltp.Parser()
    parser.load(LTP_MODEL_DIR + "/parser.model")

    srl = ltp.SementicRoleLabeller()
    srl.load(LTP_MODEL_DIR + "/pisrl.model")

    def deal(clause):

        words = segmentor.segment(clause)
        postags = postagger.postag(words)
        arcs = parser.parse(words, postags)
        roles = srl.label(words, postags, arcs)
        return words, postags, arcs, roles

    words = segmentor.segment(sent)
    postags = postagger.postag(words)
    arcs = parser.parse(words, postags)
    roles = srl.label(words, postags, arcs)
    for role in roles:
        print(role.index, words[role.index])
        for arg in role.arguments:
            print(arg.name, arg.range.start, arg.range.end, "".join(words[arg.range.start: arg.range.end + 1]))

    clause_graph = build_sentence_graph(words, postags, arcs, roles)

    plot_graph(clause_graph,
               node_label={n: "".join(["".join(words[s:e]) for s, e in d["range"]])
                           for n, d in clause_graph.nodes(data=True)},
               edge_label={(u, v): d["tag"] if "tag" in d else "" for u, v, d in clause_graph.edges(data=True)},
               node_color=[
                   "red" if "nt" in {p for s, e in d["range"] for p in postags[s:e]} else "yellow"
                   for n, d in clause_graph.nodes(data=True)
               ])

    return


def test_announcement(anno_id):
    segmentor = ltp.Segmentor()
    segmentor.load_with_lexicon(LTP_MODEL_DIR + "/cws.model", LTP_MODEL_DIR + "/lexicon")

    postagger = ltp.Postagger()
    postagger.load_with_lexicon(LTP_MODEL_DIR + "/pos.model", LTP_MODEL_DIR + "/postag_lexicon")

    parser = ltp.Parser()
    parser.load(LTP_MODEL_DIR + "/parser.model")

    srl = ltp.SementicRoleLabeller()
    srl.load(LTP_MODEL_DIR + "/pisrl.model")

    story = load_story_from_announcement_id(anno_id)

    lines = line_return_preprocess(story)
    sentences = ut.filter_list(lambda s: len(s),
                               map(lambda s: re.sub(r"【.*】", "", s).strip(),
                                   ut.flatten([line.split("。") for line in lines])))
    clause_graphs = []
    nested_clause_names = []
    nested_clauses = []
    statis = {}
    for i, sent in enumerate(sentences):
        words = segmentor.segment(sent)
        postags = postagger.postag(words)
        arcs = parser.parse(words, postags)
        roles = srl.label(words, postags, arcs)
        clause_graph = build_sentence_graph(words, postags, arcs, roles)

        clause_graphs.append(clause_graph)
        nested_clause_names.append([n for n in clause_graph])
        nested_clauses.append([ut.flatten([list(words[s:e]) for s, e in d["range"]])
                               for n, d in clause_graph.nodes(data=True)])

    node_clusters = clause_similarity(nested_clauses, nested_clause_names, stop_list=stop_list,
                                      percentile=0.95)
    graph = fuse_clause_graphs(clause_graphs, node_clusters,
                               node_merger=node_merger_wrapper(nested_clause_names, nested_clauses))
    statis.update({
        "graphs_before_fuse": len(nested_clauses),
        "node_before_fuse"  : sum([len(l) for l in nested_clauses]),
        "node_after_fuse"   : len(graph.nodes),
        "edge_after_fuse"   : len(graph.edges)
    })

    plot_graph(graph,
               node_label={n: "\n".join([cl for cl in d["clause"]])
                           for n, d in graph.nodes(data=True)},
               edge_label={(u, v): str(d["weight"]) if "weight" in d else None for u, v, d in
                           graph.edges(data=True)})

    return


def view_all_announcement():
    segmentor = ltp.Segmentor()
    segmentor.load_with_lexicon(LTP_MODEL_DIR + "/cws.model", LTP_MODEL_DIR + "/lexicon")

    postagger = ltp.Postagger()
    postagger.load_with_lexicon(LTP_MODEL_DIR + "/pos.model", LTP_MODEL_DIR + "/postag_lexicon")

    parser = ltp.Parser()
    parser.load(LTP_MODEL_DIR + "/parser.model")

    srl = ltp.SementicRoleLabeller()
    srl.load(LTP_MODEL_DIR + "/pisrl.model")
    story_codes = ["1101.TW", "2323.TW", "9910.TW", "3536.TW", "2850.TW"]

    # metas = ut.flatten([load_efficient_metas(story_code) for story_code in story_codes])
    metas = load_efficient_metas()
    print("Load metas finished")

    stories = load_story_from_metas(metas).values()
    print("Load stories finished")

    lines = ut.flatten([line_return_preprocess(story) for story in stories])
    print("Preprocess lines finished")

    sentences = filter(lambda s: len(s),
                       map(lambda s: re.sub(r"【.*】", "", s).strip(),
                           ut.flatten([line.split("。") for line in lines])))

    sentences = [segmentor.segment(sent) for sent in sentences]
    postags = [postagger.postag(sent) for sent in sentences]
    print("Segment sentences finished:", len(sentences))

    with open(LTP_MODEL_DIR + "/stoplist", "r", encoding="utf-8") as f:
        stoplist = {w.strip() for w in f.readlines()}

    def filter_func(w, i, j):
        if w in stoplist: return False
        if re.match(r"^[a-zA-Z0-9_,.]+$", w): return False
        ps = postags[i][j]
        if ps.startswith("n") or ps == "v": return True
        return False

    sentences = [[w for j, w in enumerate(s) if filter_func(w, i, j)]
                 for i, s in enumerate(sentences)]

    sentences = [s for s in sentences if len(s)]
    print("Filter sentences finished:", len(sentences))

    train_clause_representation_model(sentences)

    # with open(LTP_MODEL_DIR+"/coord_conjs", "r", encoding="UTF-8") as f:
    #     coord_conjs = {l.strip() for l in f.readlines()}
    #
    # conj = dict()
    # for sent in sentences:
    #
    #     sent = re.sub(r"【.*】", "", sent)
    #
    #     words = segmentor.segment(sent)
    #     postags = postagger.postag(words)
    #     arcs = parser.parse(words, postags)
    #     # roles = srl.label(words, postags, arcs)
    #
    #     comma = {"，", "；"}
    #     start, end = ut.split_list(words, lambda _, w: w in comma)
    #     # create nodes
    #     for i, (s, e) in enumerate(zip(start, end)):
    #         if words[s] in coord_conjs:
    #             if words[s] not in conj:
    #                 conj[words[s]] = [0, 0]
    #
    #             if "SBV" in {a.relation for a in arcs[s:e]}:
    #
    #                 conj[words[s]][0] += 1
    #             else:
    #                 print(" ".join([
    #                     "(%i)" % i+(("{"+w+"}") if w in coord_conjs else w)+"/"+a.relation+":"+str(a.head-1-s)
    #                     for i, (w, a) in enumerate(zip(words[s:e], arcs[s:e]))
    #                 ]))
    #                 conj[words[s]][1] += 1
    #
    # for k, v in sorted(conj.items(), key=lambda tpl: tpl[1], reverse=True):
    #     print(k, v[0], v[1])
    return


def compound_logic_graph(nodes, compound_type):
    if len(compound_type) == 0:
        logic_graph = nx.DiGraph()
        logic_graph.add_edges_from([("Start", "End")])
    elif len(compound_type) == 1:
        logic_graph = nx.DiGraph()
        logic_graph.add_edges_from([("Start", nodes[0]), (nodes[0], "End")])
    else:
        if compound_type[-1] == "n":
            logic_graph = compound_logic_graph(nodes[:-1], compound_type[:-1])
            preds = list(logic_graph.pred["End"])
            logic_graph.add_edges_from([(s, nodes[-1]) for s in preds])
            logic_graph.remove_edges_from([(s, "End") for s in preds])
            logic_graph.add_edge(nodes[-1], "End")
        elif compound_type[-1] == "o":
            logic_graph = compound_logic_graph(nodes[:-1], compound_type[:-1])
            logic_graph.add_edges_from([("Start", nodes[-1]), (nodes[-1], "End")])
        elif compound_type[-1] == "i":
            i = len(nodes) - 1
            while compound_type[i] == "i":
                i -= 1
            logic_graph = compound_logic_graph(nodes[:i], compound_type[:i])
            if compound_type[i] == "n":
                preds = list(logic_graph.pred["End"])
                logic_graph.add_edges_from([(s, e) for s, e in product(preds, nodes[i:])])
                logic_graph.remove_edges_from([(s, "End") for s in preds])
                logic_graph.add_edges_from([(e, "End") for e in nodes[i:]])

            elif compound_type[i] == "o":
                logic_graph.add_edges_from([("Start", s) for s in nodes[i:]])
                logic_graph.add_edges_from([(s, "End") for s in nodes[i:]])

    return logic_graph


def test_sim():
    nested_clause_names = [[0, 1, 5, 4], [7, 8, 9, 10, 11], [0, 1, 2, 7, 8], [3], [0, 3, 5, 6], [0, 1, 2], [3, 4],
                           [3, 2]]
    nested_clauses = [[['台泥', '(', '1101', ')', '、', '亚泥', '(', '1102', ')', '西进', '中国', '有', '成'],
                       ['中国', '水泥网', '“', '2017年', '中国', '水泥', '熟料', '百强榜', '”', '出炉'],
                       ['台泥', '在', '以', '产能', '5', ',', '446.7万', '公', '吨', '，', '跻身', '前', '六', '强'],
                       ['亚泥', '则', '以', '2', ',', '66', '2.9万', '公吨', '排', '第十']],
                      [['农历', '春节', '前', '赶工', '需求'], ['中国', '水泥', '价格', '持续', '喊', '涨'],
                       ['台泥', '、', '亚泥', '及', '信大', '(', '1109', ')', '去年', '12月', '销售达', '全年', '高峰'],
                       ['将', '一路旺', '到', '今年', '1月', '底', '、', '2月', '初'], ['推波', '营收', '、', '获利', '进一步', '冲高']],
                      [['2017年', '是', '水泥业', '丰收', '的', '一', '年'],
                       ['由于', '中国', '持续', '执行', '去', '产能', '、', '强化', '错峰', '生产', '和', '环保', '及', '淘汰', '低阶', '水泥', '等',
                        '措施'],
                       ['预估', '2018年', '营运', '有', '机会', '维持', '高峰'], ['赶', '工', '旺季', '发威'],
                       ['而', '农历', '春节', '落', '在', '今年', '2月', '中旬', '，', '农历', '年前', '营运', '持续', '看增']],
                      [['根据', '中国', '水泥网', '统计', '数据', '，', '全', '中国', 'P42.5', '散装', '水泥', '每', '公', '吨', '均价', '已',
                        '超过', '人民币', '400', '元', '，', '远', '超过', '近', '16', '年', '均价', '近', '人民币', '百', '元']],
                      [['台泥', '在', '长三角', '供不应求'], ['并', '延续', '到', '2018年', '1月', '、', '甚至', '2月', '上旬'],
                       ['两', '广', '地区', '市场', '出', '货价', '逐步', '增温'], ['2017年', '12月', '销售', '可望', '达', '2017年', '高峰']],
                      [['台泥', '国际', '已', '从', '港股', '完成', '下市', '私有化'],
                       ['从', '12月', '起', '台泥', '在', '大陆', '市场', '的', '营收', '获利', '可', '100%', '全数', '贡献', '给', '台泥'],
                       ['预期', '去年', '第四', '季', '营收', '、', '获利', '有', '机会', '同步', '冲高']],
                      [['主力', '江西', '、', '湖北', '市场', '水泥', '价格', '补涨'],
                       ['亚泥', '第四', '季', '营运', '也', '有', '机会', '进一步', '冲高']],
                      [['“', '2017年', '中国', '水泥', '熟料', '百强榜', '”', '是', '以', '大陆', '水泥', '企业', '去年', '熟料', '产能',
                        '所', '作', '的', '排行',
                        '，', '中国', '建材', '集团', '、', '海螺', '水泥', '、', '金隅', '冀东', '集团', '分列', '前', '三', '名'],
                       ['第4', '到', '第10', '名', '分别', '为', '华润', '、', '华新', '拉法基', '、', '台泥', '、', '山水', '、', '红狮',
                        '、', '天瑞', '及',
                        '亚泥']]]

    matching = clause_similarity(nested_clauses, nested_clause_names, stop_list, percentile=0.9)
    for i in matching:
        print(i)
        for t in i:
            x, y = t
            n = ut.find(nested_clause_names[x], lambda _, ii: ii == y)
            print("".join(nested_clauses[x][n]))
        print()
    return


if __name__ == '__main__':
    a = 0
