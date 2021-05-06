import json
import os
from queue import Queue

import networkx as nx
import numpy as np

import utils as ut
from config import ABSTRACT_GRAPH_DIR, ABSTRACT_NODE_MAPPING_DIR, ANNOCOM_FEATURE_DIR, HEADLINE_NODE_MAPPING_DIR
from text.abstract_graph.feature_utils import load_event_feat, load_vocab_mappings
from predict.conf import AVERAGE_POOLING


def extract_event_embedding(beta=0.8):
    """
    This function will calculate event embedding matrix. One event will be embedded into a K-dimensional vector. K is
    the number of events. The j-th element in vector-i is like the distance between j-th event and i-th event.
    `beta` denoted all successors will receive bete proportion of weight in root node.
    :param beta:
    :return:
    """
    with open(ABSTRACT_GRAPH_DIR + "/abstract_graph.json", "r", encoding="utf-8") as f:
        abstract_graph = json.load(f)
        abstract_graph = nx.node_link_graph(abstract_graph)

    assert isinstance(abstract_graph, nx.MultiDiGraph)

    w2i, i2w = load_vocab_mappings()
    id2event, event_feats = load_event_feat((w2i, i2w))
    E, D = len(id2event), len(id2event)
    event_features = np.zeros((E, D))

    for node in abstract_graph.nodes:
        neighbors = abstract_graph.adj[node]
        # sum_of_weights = sum([dct[0]["weight"] for dct in neighbors.values()])
        for neighbor, dct in neighbors.items():
            # propagate proportion
            abstract_graph.edges[node, neighbor, 0]["normalized_weight"] = dct[0]["weight"]/(
                np.sqrt((abstract_graph.nodes[node]["cnt"] + 1)*(abstract_graph.nodes[neighbor]["cnt"] + 1))
            )

    with open(os.path.join(ABSTRACT_GRAPH_DIR, "abstract_graph_normalized.json"), "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(abstract_graph), f, indent=4, ensure_ascii=False)

    em = {n: int(n) - 1 for n in abstract_graph.nodes}
    for node in abstract_graph.nodes:
        q = Queue()
        visited = set()
        event_features[em[node], em[node]] = 1.0
        q.put(node)
        while q.qsize() > 0:
            n = q.get()
            w = event_features[em[node], em[n]]
            if w < 1e-2: continue
            visited.add(n)

            if len(abstract_graph.succ[n]) == 0: continue
            norm_inf = np.linalg.norm([dct[0]["normalized_weight"] for succ, dct in abstract_graph.succ[n].items()],
                                      ord=np.inf)
            for succ, dct in abstract_graph.succ[n].items():
                if succ in visited: continue
                q.put(succ)
                visited.add(succ)
                # weight formula
                # event_features[em[node], em[succ]] = w*beta*np.exp(
                #     dct[0]["normalized_weight"] - 1/len(abstract_graph.succ[n]))

                event_features[em[node], em[succ]] = w*beta*(dct[0]["normalized_weight"]/norm_inf)

    np.save(os.path.join(ANNOCOM_FEATURE_DIR, "event_features.npy"), event_features)
    return True


def extract_annocom_feature(node_mapping_mode="full-text"):
    """
    This function is used to represent an announcement into a feature vector. All abstract events in this
    announcements will be mapped to event embedding and applied sum/mean pooling.

    NOTE: `announcement` in variable name means "announcement_id:company_code" format string.
    :return:
    """
    print("Start to extract announcement features.")

    event_features = np.load(os.path.join(ANNOCOM_FEATURE_DIR, "event_features.npy"))
    E, D = event_features.shape

    annocom2identifier = {}

    # bow represent announcement identifier
    if node_mapping_mode == "full-text":
        directory = ABSTRACT_NODE_MAPPING_DIR
    else:
        directory = HEADLINE_NODE_MAPPING_DIR

    for filename in ut.filter_list(lambda f: f.endswith("json"), os.listdir(directory)):
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            announcement_event_mappings = json.load(f)

        if node_mapping_mode == "full-text":
            for announcement, com2node_mappings in announcement_event_mappings.items():
                for com, node_mappings in com2node_mappings.items():
                    annocom2identifier[announcement + ":" + com] = np.zeros((E,))
                    events = set(node_mappings.values())
                    annocom2identifier[announcement + ":" + com][
                        [int(e) - 1 for e in events]
                    ] = (1.0/len(events)) if AVERAGE_POOLING else 1.0
        else:
            for announcement, com2events in announcement_event_mappings.items():
                for com, events in com2events.items():
                    annocom2identifier[announcement + ":" + com] = np.zeros((E,))
                    events = set(events)
                    annocom2identifier[announcement + ":" + com][
                        [int(e) - 1 for e in events]
                    ] = (1.0/len(events)) if AVERAGE_POOLING else 1.0

    print("Finish representing announcement into BOW format.")

    # stack announcement_feature in some order
    sorted_annocom = sorted(annocom2identifier)
    with open(os.path.join(ANNOCOM_FEATURE_DIR, "featured_annocoms.txt"), "w", encoding="utf-8") as f:
        f.writelines(["{}\n".format(annocom) for annocom in sorted_annocom])

    annocom_identifier = np.vstack([
        annocom2identifier[annocom] for annocom in sorted_annocom
    ])
    print("Finish stacking announcement identifiers in {} matrix.".format(annocom_identifier.shape))

    # save announcement_feature
    np.save(os.path.join(ANNOCOM_FEATURE_DIR, "annocom_feature.npy"),
            np.matmul(annocom_identifier, event_features))
    print("All announcement features are written in {}.".format("annocom_feature.npy"))
    return True


def annocom_event_distribution():
    print("Start to extract announcement features.")

    event_features = np.load(os.path.join(ANNOCOM_FEATURE_DIR, "event_features.npy"))
    E, D = event_features.shape

    w2i, i2w = load_vocab_mappings()
    id2event, event_feats = load_event_feat((w2i, i2w))

    annocom2events = {}

    directory = HEADLINE_NODE_MAPPING_DIR

    for filename in ut.filter_list(lambda f: f.endswith("json"), os.listdir(directory)):
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            announcement_event_mappings = json.load(f)

        for announcement, com2events in announcement_event_mappings.items():
            for com, events in com2events.items():
                annocom2events[announcement + ":" + com] = set(events)
    event2annocoms = ut.group_by_key([
        (annocom, event) for annocom, events in annocom2events.items() for event in events
    ], key=lambda t: t[1], value=lambda t: t[0])

    print("Total Announcement: {}".format(len(annocom2events)))
    for i in range(E):
        real_i = str(i + 1)
        real_event = id2event[str(i + 1)]
        print("Event {}: {};  Annocom: {}; Annocom_prob: {}".format(
            real_i, real_event, len(event2annocoms[real_i]), round(len(event2annocoms[real_i])/len(annocom2events), 4))
        )

    return


if __name__ == '__main__':
    extract_event_embedding()
