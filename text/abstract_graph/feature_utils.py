import pickle
import pyltp as ltp

import utils as ut
from config import ABSTRACT_VOCAB_DIR, LTP_MODEL_DIR
from collections import defaultdict

import networkx as nx
from itertools import combinations, chain
from queue import Queue

import os


def load_vocab_mapping(filename):
    with open(ABSTRACT_VOCAB_DIR+"/"+filename, "r", encoding="utf-8") as f:
        key_words = [l.strip().split() for l in f.readlines()]
        word2id = {tpl[1]: tpl[0] for tpl in key_words}
        id2word = ut.group_by_key(word2id, key=lambda k: word2id[k], value=lambda k: k)
    return word2id, id2word


def load_vocab_mappings():
    w2i = {}
    i2w = {}
    w2i["general_subject"], i2w["general_subject"] = load_vocab_mapping("general_subject_cluster")
    w2i["verb"], i2w["verb"] = load_vocab_mapping("verb_cluster")
    w2i["geographical"], i2w["geographical"] = load_vocab_mapping("geographical_cluster")
    # w2i["negative"], i2w["negative"] = load_vocab_mapping("negative")
    # w2i["positive"], i2w["positive"] = load_vocab_mapping("positive")

    return w2i, i2w


def load_event_feat(vocab_mappings):
    with open(ABSTRACT_VOCAB_DIR+"/abstract_event_taxonomy", "r", encoding="utf-8") as f:
        event_lines = [l.strip().split() for l in f.readlines()]

    event_lines = [(l[0], l[1], l[2].split("、"), l[3].split("、") if len(l) > 3 else []) for l in event_lines]

    w2i, i2w = vocab_mappings

    id2event = {l[0]: l[1] for l in event_lines}
    id2feat = {l[0]: {k: {w2i[k][w] for w in chain(l[2], l[3]) if w in w2i[k]} for k in w2i}
               for l in event_lines}

    id2must = {l[0]: [
        {k: {w2i[k][w] for w in ws.split("|") if w in w2i[k]} for k in w2i} for ws in l[3]
    ] for l in event_lines}

    event_feats = {id: {
        "feature": id2feat[id],
        "must"   : id2must[id],
        "length" : sum([len(feats) for feats in id2feat[id].values()]),
    } for id in id2feat}

    return id2event, event_feats


def feature_word_extractor_wrapper(filename):
    word2id, _ = load_vocab_mapping(filename)

    return lambda word_set: [(word2id[w], w) for w in word_set if w in word2id]


def flatten_feature_word_synset(ifilename, ofilename):
    with open(os.path.join(ABSTRACT_VOCAB_DIR, ifilename), "r", encoding="utf-8") as f:
        key_words = [l.strip() for l in f.readlines() if len(l.strip()) != 0]

    with open(os.path.join(ABSTRACT_VOCAB_DIR, ofilename), "w", encoding="utf-8") as f:
        f.writelines([
            "{}\t{}\n".format(i, w)
            for i, kw in enumerate(key_words) for w in kw.split()
        ])
    return True


class FeatureUtils:
    def __init__(self):
        self.segmentor = ltp.Segmentor()
        self.segmentor.load_with_lexicon(LTP_MODEL_DIR+"/cws.model", LTP_MODEL_DIR+"/lexicon")

        self.postagger = ltp.Postagger()
        self.postagger.load_with_lexicon(LTP_MODEL_DIR+"/pos.model", LTP_MODEL_DIR+"/postag_lexicon")

        self.base_general_subject_extractor = feature_word_extractor_wrapper("general_subject_cluster")
        self.verb_extractor = feature_word_extractor_wrapper("verb_cluster")
        self.geographical_extractor = feature_word_extractor_wrapper("geographical_cluster")
        self.positive_extractor = feature_word_extractor_wrapper("positive")
        self.negative_extractor = feature_word_extractor_wrapper("negative")

        with open(ABSTRACT_VOCAB_DIR+"/tfidf.model", "rb") as f:
            self.tfidf_model = pickle.load(f)
            self.idf_lower_bound = 0.9 * max(self.tfidf_model.idf_)+0.1 * min(self.tfidf_model.idf_)

        with open(LTP_MODEL_DIR+"/stoplist", "r", encoding="utf-8") as f:
            self.stop_list = {w.strip() for w in f.readlines()}

    def __low_df__(self, w):
        if w in self.stop_list or len(w) == 1: return False

        if w in self.tfidf_model.vocabulary_:
            return self.tfidf_model.idf_[self.tfidf_model.vocabulary_[w]] > self.idf_lower_bound
        else:

            return self.postagger.postag([w])[0] != "m"

    def update_name_entity(self, sentences):
        words = [list(self.segmentor.segment(sent)) for sent in sentences]
        low_df_words = [(w, (i, j)) for i, sent in enumerate(words)
                        for j, w in enumerate(sent) if self.__low_df__(w)]
        word2loc = ut.group_by_key(low_df_words, key=lambda t: t[0], value=lambda t: t[1])
        ne2subject = {w: [
            tpl for l in ls
            for tpl in self.base_general_subject_extractor(set(words[l[0]][max(l[1]-5, 0):l[1]+5]))
        ] for w, ls in word2loc.items()}
        return ne2subject

    def name_entity_extractor(self, word_set, ne2subject):
        return [w for w in word_set if w in ne2subject]

    def general_subject_extractor(self, word_set):
        return self.base_general_subject_extractor(word_set)


if __name__ == '__main__':
    flatten_feature_word_synset("verb", "verb_cluster")
    flatten_feature_word_synset("general_subject", "general_subject_cluster")
