import json
import os
import pyltp as ltp
import re
import traceback
from collections import defaultdict

import networkx as nx
from progressbar import ProgressBar

import utils as ut
from text.concrete_graph.clause_similarity import clause_similarity, fuse_clause_graphs, node_merger_wrapper
from text.concrete_graph.paratactic_sentence_graph import build_sentence_graph, plot_graph
from config import CONCRETE_GRAPH_DIR, CONCRETE_PLOT_DIR, LTP_MODEL_DIR
from database_model.loader import load_efficient_metas, load_story_from_metas
from preprocess import line_return_preprocess


# 判断是否为事实句,没有用到,可以忽略
def fact_graph(clause_graph):
    return len(clause_graph.nodes) <= 3 and all(["tag" not in d for u, v, d in clause_graph.edges(data=True)])


# 保存事理图谱为json格式
def save_graph(id2graphs: dict, id2sentences: dict, id2statis: dict, filename: str):
    graph_datas = {id: {
        "graph"   : nx.node_link_data(graph),
        "sentence": id2sentences[id],
        "statis"  : id2statis[id],
    } for id, graph in id2graphs.items()}
    with open(CONCRETE_GRAPH_DIR + "/concrete-" + filename + ".json", "w", encoding="utf-8") as f:
        json.dump(graph_datas, f, indent=4, ensure_ascii=False)
    return True


# 入口函数
def build_announcement_graph():
    segmentor = ltp.Segmentor()
    segmentor.load_with_lexicon(LTP_MODEL_DIR + "/cws.model", LTP_MODEL_DIR + "/lexicon")

    postagger = ltp.Postagger()
    postagger.load_with_lexicon(LTP_MODEL_DIR + "/pos.model", LTP_MODEL_DIR + "/postag_lexicon")

    parser = ltp.Parser()
    parser.load(LTP_MODEL_DIR + "/parser.model")

    srl = ltp.SementicRoleLabeller()
    srl.load(LTP_MODEL_DIR + "/pisrl.model")

    with open(LTP_MODEL_DIR + "/stoplist", "r", encoding="utf-8") as f:
        stop_list = {w.strip() for w in f.readlines()}

    metas = load_efficient_metas()

    id2metas = {meta.announcement_id: meta for meta in metas}
    filedate2id = ut.group_by_key({id: id.split("-")[0] for id in id2metas}.items(),
                                  key=lambda tpl: tpl[1], value=lambda tpl: tpl[0])

    print("Load metas finished. There are {} files and {} announcements in total.".format(len(filedate2id),
                                                                                          len(id2metas)))
    # id 指 announcement id
    for filename, ids in list(filedate2id.items()):
        print("Processing filename: {}, there are {} announcements in this file.".format(filename, len(ids)))

        id2stories = load_story_from_metas([id2metas[id] for id in ids])
        id2graphs = {}
        id2sentences = {}
        id2statis = {}

        progress = ProgressBar()
        for id, story in progress(sorted(id2stories.items(), key=lambda tpl: tpl[0])):
            try:
                id2statis[id] = defaultdict(int)
                lines = line_return_preprocess(story)
                sentences = ut.filter_list(lambda s: len(s),
                                           map(lambda s: re.sub(r"【.*】", "", s).strip(),
                                               ut.flatten([line.split("。") for line in lines])))
                id2sentences[id] = sentences
                clause_graphs = []
                nested_clause_names = []  # 双层列表,表示每句话中的clause-id
                nested_clauses = []     # 双层列表,表示每句话中的clause内容
                for i, sent in enumerate(sentences):
                    # 预处理
                    words = segmentor.segment(sent)
                    postags = postagger.postag(words)
                    arcs = parser.parse(words, postags)
                    roles = srl.label(words, postags, arcs)

                    # 创建句子级事理图谱
                    clause_graph = build_sentence_graph(words, postags, arcs, roles)
                    # print("".join(words))
                    if fact_graph(clause_graph):
                        id2statis[id]["fact_sentence"] += 1

                    clause_graphs.append(clause_graph)
                    nested_clause_names.append([n for n in clause_graph])
                    nested_clauses.append([ut.flatten([list(words[s:e]) for s, e in d["range"]])
                                           for n, d in clause_graph.nodes(data=True)])

                if len(clause_graphs) == 0:
                    print("Announcement {} contains no clause graphs.".format(id))
                    continue
                # 计算得到每一个node cluster,双层列表,每一个元素表示一个cluster集合,集合中为每个node的id
                node_clusters = clause_similarity(nested_clauses, nested_clause_names, stop_list=stop_list,
                                                  percentile=0.95)
                id2graphs[id] = fuse_clause_graphs(clause_graphs, node_clusters,
                                                   node_merger=node_merger_wrapper(nested_clause_names, nested_clauses))
                id2statis[id].update({
                    "graphs_before_fuse": len(nested_clauses),
                    "node_before_fuse"  : sum([len(l) for l in nested_clauses]),
                    "node_after_fuse"   : len(id2graphs[id].nodes),
                    "edge_after_fuse"   : len(id2graphs[id].edges)
                })

                plot_graph(id2graphs[id],
                           node_label={n: "\n".join([cl for cl in d["clause"]])
                                       for n, d in id2graphs[id].nodes(data=True)},
                           edge_label={(u, v): str(d["weight"]) if "weight" in d else None for u, v, d in
                                       id2graphs[id].edges(data=True)},
                           filename=CONCRETE_PLOT_DIR + "/{}.png".format(id))
            except Exception as e:
                print("An error occurs when processing announcement {}".format(id))
                traceback.print_exc()

        save_graph(id2graphs, id2sentences, id2statis, filename)
        print("Finish file-date {}".format(filename))
    return True


if __name__ == '__main__':
    build_announcement_graph()
