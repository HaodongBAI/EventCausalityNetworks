import json
import os
import pyltp as ltp
import random
from collections import Counter

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import utils as ut
from config import ABSTRACT_VOCAB_DIR, CONCRETE_GRAPH_DIR, LTP_MODEL_DIR


def build_word2vec_model(extend_mode="doc", n_files=200):
    segmentor = ltp.Segmentor()
    segmentor.load_with_lexicon(LTP_MODEL_DIR+"/cws.model", LTP_MODEL_DIR+"/lexicon")

    corpus = []
    filenames = ut.filter_list(lambda f: f.endswith("json"), os.listdir(CONCRETE_GRAPH_DIR))
    for filename in random.sample(filenames, n_files):

        with open(os.path.join(CONCRETE_GRAPH_DIR, filename), "r", encoding="utf-8") as f:
            id2concrete = json.load(f)

        if extend_mode == "doc":
            corpus.extend(
                [[w for sent in concrete["sentence"]
                  for w in segmentor.segment(sent)]
                 for _, concrete in id2concrete.items()]
            )
        else:  # sent level
            corpus.extend(
                [[w for w in segmentor.segment(sent)]
                 for _, concrete in id2concrete.items()
                 for sent in concrete["sentence"]]
            )

    w2v = Word2Vec(corpus, size=100, window=5, min_count=10, workers=4)
    w2v.save(ABSTRACT_VOCAB_DIR+"/word2vec.model")
    return True


def build_tfidf_model(n_files=200):
    segmentor = ltp.Segmentor()
    segmentor.load_with_lexicon(LTP_MODEL_DIR+"/cws.model", LTP_MODEL_DIR+"/lexicon")

    with open(LTP_MODEL_DIR+"/stoplist", "r", encoding="utf-8") as f:
        stop_list = {w.strip() for w in f.readlines()}

    corpus = []
    filenames = ut.filter_list(lambda f: f.endswith("json"), os.listdir(CONCRETE_GRAPH_DIR))
    for filename in random.sample(filenames, n_files):
        with open(os.path.join(CONCRETE_GRAPH_DIR, filename), "r", encoding="utf-8") as f:
            id2concrete = json.load(f)

        corpus.extend([
            " ".join([w for sent in concrete["sentence"]
                      for w in segmentor.segment(sent)])
            for _, concrete in id2concrete.items()]
        )

    tfidf = TfidfVectorizer(stop_words=stop_list)
    tfidf.fit(corpus)
    with open(ABSTRACT_VOCAB_DIR+"/tfidf.model", "wb") as f:
        pickle.dump(tfidf, f, -1)
    return True


def cluster_words(ifilename, ofilename, k):
    with open(ifilename, "r", encoding="utf-8") as f:
        words = [[w.strip()] for w in f.readlines()]
    dct = Dictionary(words)

    print("Build a Dictionary with {} entries from {} words.".format(len(dct), len(words)))
    w2v = Word2Vec.load(ABSTRACT_VOCAB_DIR+"/word2vec.model")

    dct.filter_tokens(bad_ids=[
        w for w in dct.token2id if w not in w2v.wv.vocab
    ])
    w2v.init_sims(replace=True)
    sim = w2v.wv.similarity_matrix(dct)

    # plt.hist([i for i in sim.toarray().flatten() if 0.05 < i < 0.95], bins=200)
    # plt.show()
    # inner_prod = lambda w1, w2: np.dot(X[dct.token2id[w1], :], X[dct.token2id[w2], :])
    # euc_dist = lambda w1, w2: np.sqrt(np.sum(np.square(X[dct.token2id[w1], :]-X[dct.token2id[w2], :])))
    # similar = lambda w1, w2: sim[dct.token2id[w1], dct.token2id[w2]]
    # print(w2v.wv.similarity("兼具", "对应"), inner_prod("兼具", "对应"), euc_dist("兼具", "对应"), similar("兼具", "对应"))
    # print(w2v.wv.similarity("兼具", "感知"), inner_prod("兼具", "感知"), euc_dist("兼具", "感知"), similar("兼具", "感知"))
    # print(w2v.wv.similarity("兼具", "接取"), inner_prod("兼具", "接取"), euc_dist("兼具", "接取"), similar("兼具", "接取"))
    # print(w2v.wv.similarity("兼具", "播放"), inner_prod("兼具", "播放"), euc_dist("兼具", "播放"), similar("兼具", "播放"))
    # print(w2v.wv.similarity("兼具", "撷取"), inner_prod("兼具", "撷取"), euc_dist("兼具", "撷取"), similar("兼具", "撷取"))
    # print(w2v.wv.similarity("兼具", "识别"), inner_prod("兼具", "识别"), euc_dist("兼具", "识别"), similar("兼具", "识别"))
    # print(w2v.wv.similarity("兼具", "追踪"), inner_prod("兼具", "追踪"), euc_dist("兼具", "追踪"), similar("兼具", "追踪"))

    model = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete')
    cls = model.fit_predict(1-sim.toarray())

    counts = Counter(cls)
    for i, (c, cnt) in enumerate(sorted(counts.items(), key=lambda t: t[1], reverse=True)[:10]):
        print("{}:: {}. Cluster {} has {} words.".format(k, i, c, cnt))

    with open(ofilename, "w", encoding="utf-8") as f:
        f.writelines(["{}\t{}\n".format(c, i) for c, i in sorted([
            (c, dct.id2token[i]) for i, c in enumerate(cls)
        ])])
    return


if __name__ == '__main__':
    # cluster_words(
    #     ABSTRACT_VOCAB_DIR+"/negative",
    #     ABSTRACT_VOCAB_DIR+"/negative_cluster",
    #     k=400)
    build_tfidf_model()
