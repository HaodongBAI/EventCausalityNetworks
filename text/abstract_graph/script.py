import json
import os
import pyltp as ltp
import random
from collections import defaultdict
from functools import reduce

import utils as ut
from config import ABSTRACT_VOCAB_DIR, CONCRETE_GRAPH_DIR, CONCRETE_INVERTED_JSON_DIR, CONCRETE_NODEFEAT_JSON_DIR, \
    LTP_MODEL_DIR
from text.abstract_graph.feature_utils import load_vocab_mappings
from text.abstract_graph.node_event_mapping import jaccard_similar_func


def view_all_concrete():
    segmentor = ltp.Segmentor()
    segmentor.load_with_lexicon(LTP_MODEL_DIR+"/cws.model", LTP_MODEL_DIR+"/lexicon")

    postagger = ltp.Postagger()
    postagger.load_with_lexicon(LTP_MODEL_DIR+"/pos.model", LTP_MODEL_DIR+"/postag_lexicon")

    def helper(r2c, r):
        return reduce(lambda s, t: (min(s[0], t[0]), max(s[1], t[1])), [
            helper(r2c, c) for c in r2c[r]
        ], (r, r+1))

    filenames = ut.filter_list(lambda f: f.endswith("json"), os.listdir(CONCRETE_GRAPH_DIR))
    j = 0
    ws = set()
    for filename in random.sample(filenames, 200):
        print(j)
        j += 1
        with open(os.path.join(CONCRETE_GRAPH_DIR, filename), "r", encoding="utf-8") as f:
            id2concrete = json.load(f)

        for id, concrete in id2concrete.items():
            sentence = concrete["sentence"]
            for sent in sentence:
                words = segmentor.segment(sent)
                for word in words:
                    if word.endswith("业"):
                        ws.add(word)

    for w in ws:
        print(w)

    return


def view_anno():
    segmentor = ltp.Segmentor()
    segmentor.load_with_lexicon(LTP_MODEL_DIR+"/cws.model", LTP_MODEL_DIR+"/lexicon")

    postagger = ltp.Postagger()
    postagger.load_with_lexicon(LTP_MODEL_DIR+"/pos.model", LTP_MODEL_DIR+"/postag_lexicon")

    with open("/Users/Kevin/Documents/Develop/PyCharmPython/twquant/text/temp/anno", "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    words = [segmentor.segment(l) for l in lines]
    postags = [postagger.postag(ws) for ws in words]
    nouns = ut.flatten(
        [
            [w for w, p in zip(ws, ps) if p == "n"]
            for ws, ps in zip(words, postags)
        ]
    )
    for n in set(nouns):
        print(n)
    return


def dealwith_subject():
    with open(ABSTRACT_VOCAB_DIR+"/general_subject", "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    s, e = ut.split_list(lines, lambda _, l: len(l) == 0)
    tfidf = [l.split() for l in lines[s[0]:e[0]]]
    vocab = set(lines[s[1]:e[1]])
    for w, p, idf in tfidf:
        if w in vocab:
            continue
        else:
            print(w)


def build_inverted_file():
    """
    inverted := { key: feat_id2node:={ feat_id: nodes:=[ anno_node_id := announcement_id+":"+node_id ] }
    :return:
    """
    filenames = ut.filter_list(lambda f: f.endswith("json"), os.listdir(CONCRETE_NODEFEAT_JSON_DIR))

    inverted = {
        "general_subject": defaultdict(list),
        "verb"           : defaultdict(list),
        "geographical"   : defaultdict(list),
        "positive"       : defaultdict(list),
        "negative"       : defaultdict(list),
    }

    for filename in filenames:
        with open(os.path.join(CONCRETE_NODEFEAT_JSON_DIR, filename), "r", encoding="utf-8") as f:
            node_feats = json.load(f)
        print("Processing filename {}. There are {} concrete graphs.".format(filename, len(node_feats)))
        for key in inverted:
            for announcement_id, node_id, feat_id in (
                (announcement_id, node_id, feat_tpl[0])
                for announcement_id, node_feats in node_feats.items()
                for node_id, dct in node_feats.items()
                for feat_tpl in dct["feature"][key]
            ):
                inverted[key][feat_id].append(announcement_id+":"+node_id)

    with open(CONCRETE_INVERTED_JSON_DIR+"/inverted_file.json", "w", encoding="utf-8") as f:
        json.dump(inverted, f)
    return


def script_cooccurance_feat_words(queries, inverted, N=10):
    for k, w2n in inverted.items():
        for w, ns in w2n.items():
            inverted[k][w] = set(ns)

    w2i, i2w = load_vocab_mappings()

    sim = defaultdict(dict)
    for key, query_word in queries:
        if query_word not in w2i[key]:
            raise ValueError("Query word {} is not in Topic {}.".format(query_word, key))
        query = w2i[key][query_word]
        query_nodes = inverted[key][query]

        # calculate co-occurance
        for k, w in ((k, w) for k in inverted for w in inverted[k]):
            if k == key and w == query: continue
            c = len(query_nodes.intersection(inverted[k][w]))
            if c == 0: continue
            sim[(key, query)][(k, w)] = jaccard_similar_func(c, len(query_nodes), len(inverted[k][w]))

        # print clauses
        for filename in ut.filter_list(lambda f: f.endswith("json"), os.listdir(CONCRETE_NODEFEAT_JSON_DIR)):
            with open(os.path.join(CONCRETE_NODEFEAT_JSON_DIR, filename), "r", encoding="utf-8") as f:
                node_feats = json.load(f)
            with open(os.path.join(CONCRETE_GRAPH_DIR, filename.replace("nodefeat", "concrete")), "r",
                      encoding="utf-8") as f:
                id2concrete = json.load(f)

            for anno_id, node_id in ((ai, ni) for ai in node_feats for ni in node_feats[ai]):
                if query not in [tpl[0] for tpl in node_feats[anno_id][node_id]["feature"][key]]: continue
                for dct in id2concrete[anno_id]["graph"]["nodes"]:
                    if str(dct["id"]) != node_id: continue
                    print(" | ".join(dct["clause"]))
                    break

    sim = {t0: [
        (v, [w for it, w in enumerate(i2w[k][i]) if it < 3])
        for (k, i), v in sorted(t1.items(), key=lambda t: t[1], reverse=True)[:N]
    ] for t0, t1 in sim.items()}

    # print co-occurance
    for q, o in sim.items():
        print(q)
        for v, oo in o:
            print("\t", oo, "%.4f" % v)
    return sim


if __name__ == '__main__':
    view_all_concrete()

