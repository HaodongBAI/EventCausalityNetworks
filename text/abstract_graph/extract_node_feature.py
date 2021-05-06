import json
import os

import utils as ut
from config import CONCRETE_GRAPH_DIR, CONCRETE_NODEFEAT_JSON_DIR
from text.abstract_graph.feature_utils import FeatureUtils


def node_argument_recognition(concrete, feat_utils: FeatureUtils):
    """
    This function is used to extract feature words from one each node in concrete_graph. FeatureUtils contains a
    bunch of tools to extract feature words. The result is put into one generator. Feature extraction is simply
    implemented via BOW format of node clause.

    Deprecated:
    Name-Entity extraction: Originally, FeatureUtils will extract low frequency words as name-entity. Then the
    general subject related to one name-entity will be added all nodes that contain this name entity. But the NER
    perform poorly so we deprecate this function.
    :param concrete: concrete data structure. See concrete_graph part.
    :param feat_utils:
    :return:
    """
    ne2subject = feat_utils.update_name_entity(concrete["sentence"])
    for node in concrete["graph"]["nodes"]:
        node_id, clauses = node["id"], node["clause"]
        word_set = [[w for w in feat_utils.segmentor.segment(clause)] for clause in clauses]
        word_set = {w1 + w2 for ws in word_set for w1, w2 in zip(ws[:-1], ws[1:])}.union(
            {w for ws in word_set for w in ws})
        ne = feat_utils.name_entity_extractor(word_set, ne2subject)

        yield (node_id, ne, {
            "general_subject": feat_utils.general_subject_extractor(word_set),
            "verb"           : feat_utils.verb_extractor(word_set),
            "geographical"   : feat_utils.geographical_extractor(word_set),
        })


def extract_node_feature():
    """
    This function is used to extract node feature words in vocabulary.
    :return:
    """
    filenames = ut.filter_list(lambda f: f.endswith("json"), os.listdir(CONCRETE_GRAPH_DIR))
    feat_utils = FeatureUtils()

    for filename in filenames:
        with open(os.path.join(CONCRETE_GRAPH_DIR, filename), "r", encoding="utf-8") as f:
            id2concrete = json.load(f)
        print("Processing filename {}. There are {} concrete graphs.".format(filename, len(id2concrete)))

        node_feats = {
            id: {
                node_id: {
                    "name_entity": list(ne),
                    "feature"    : feat_dict,
                    "length"     : sum([len(v) for v in feat_dict.values()])
                }
                for node_id, ne, feat_dict in node_argument_recognition(concrete, feat_utils)
            }
            for id, concrete in id2concrete.items()
        }

        with open(CONCRETE_NODEFEAT_JSON_DIR + "/" + filename.replace("concrete", "nodefeat"), "w",
                  encoding="utf-8") as f:
            json.dump(node_feats, f, indent=4, ensure_ascii=False)
    return


if __name__ == '__main__':
    extract_node_feature()
