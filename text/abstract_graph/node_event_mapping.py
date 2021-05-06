import json
import os
import re
from itertools import product

import networkx as nx
import numpy as np

import utils as ut
from config import ABSTRACT_NODE_MAPPING_DIR, CONCRETE_GRAPH_DIR, CONCRETE_NODEFEAT_JSON_DIR, HEADLINE_FEATURE_DIR, \
    HEADLINE_NODE_MAPPING_DIR
from database_model.loader import load_efficient_metas
from text.abstract_graph.feature_utils import load_event_feat, load_vocab_mappings
from itertools import chain


def cosine_similar_func(c, l, r):
    return c/np.sqrt(l*r)


def jaccard_similar_func(c, l, r):
    return c/(l + r - c)


def jaccard_maximum_func(c, l, r):
    return max(c/l, c/r)


def node_event_similarity(announcement_node_feats: dict, event_feats: dict):
    """
    This function is used to calculate similarity between events and all nodes in one file. The variable Must
    are used to limit events that node can be mapped to. This condition is formulated as a Conjunctive Normal Form(CNF).


    Data Structure:
    announcement_node_feats := { announcement_id: node_feats := { node_id: {
        "feature": k2feats := { key: feats:=[ feat_tpl:=(feat_id, feat_word) ] }
        "length" : length
    } } }

    event_feats := { event_id:
        "feature": k2feats := { key: feats:=[ feat_id ] }
        "length" : length
        "must"   : [feat_dct := { key: feats:=[ feat_id ]}]
    } }

    feat2event := { key: feat_id2event := { feat_id: events:={ event_id }}

    :param announcement_node_feats: dict of node features
    :param event_feats: dict of abstract event features
    :param feat2event: dict of features
    :return:
    """
    announcement_node2event = {
        announcement_id: {node_id: [] for node_id in node_feats}
        for announcement_id, node_feats in announcement_node_feats.items()}

    for announcement_id, node_feats in announcement_node_feats.items():
        for node, event in product(node_feats, event_feats):
            if not (event_feats[event]["length"] > 0 and node_feats[node]["length"] > 0): continue

            # If node feature satisfy the CNF condition, `must` is True.
            must = all([any([
                feat in {feat_tpl[0] for feat_tpl in node_feats[node]["feature"][k]}
                for k in ["general_subject", "verb", "geographical"] for feat in feat_dct[k]
            ]) for feat_dct in event_feats[event]["must"]])

            cnt = sum([
                len(event_feats[event]["feature"][k].intersection(
                    feat_tpl[0] for feat_tpl in node_feats[node]["feature"][k]
                )) for k in event_feats[event]["feature"]
            ])
            if not must or cnt == 0: continue
            announcement_node2event[announcement_id][node].append((
                event,
                jaccard_maximum_func(cnt, node_feats[node]["length"], event_feats[event]["length"])))

    announcement_node2event = {
        announcement_id: {
            node: sorted(events, key=lambda tpl: tpl[1], reverse=True)[:5]
            for node, events in node2event.items()}
        for announcement_id, node2event in announcement_node2event.items()}

    return announcement_node2event


def event_select_rule(events):
    events = list(filter(lambda tpl: tpl[1] > 0.2, events))
    if len(events) > 0:
        return events[0][0]
    return None


def sign_event_rule(headline):
    sign_event = []
    if "▲" in headline or "△" in headline:
        sign_event.append("118")
    elif "▼" in headline or "▽" in headline:
        sign_event.append("119")

    return sign_event


def company_selector(meta):
    main_stock = meta.main_stock
    if main_stock is None:
        stock_code_self = meta.stock_code_self
        if stock_code_self is None or len(stock_code_self) == 0: return []
        stocks = stock_code_self.split(",")
        if len(stocks) == 0: return []
        return [stock for stock in stocks]
    else:
        return [main_stock.replace(".TWN", ".TW"), ]


def hot_stock_company2node(concrete_data):
    concrete_graph = nx.node_link_graph(concrete_data["graph"])
    cc = nx.weakly_connected_components(concrete_graph)
    com2node = {}
    for component in cc:
        coms = ut.flatten([
            re.findall("\((\d{4})\)", " ".join(concrete_graph.nodes[n]["clause"])) for n in component])
        if len(coms) == 0:
            continue
        else:
            com2node[coms[0] + ".TW"] = {str(n) for n in component}
    return com2node


def node_event_mapping():
    filenames = ut.filter_list(lambda f: f.endswith("json"), os.listdir(CONCRETE_NODEFEAT_JSON_DIR))

    metas = load_efficient_metas()
    announcement2tag = {meta.announcement_id: meta.tag for meta in metas}
    announcement2company = ut.filter_dict(lambda k, v: len(v) > 0,
                                          {meta.announcement_id: company_selector(meta) for meta in metas})

    w2i, i2w = load_vocab_mappings()
    id2event, event_feats = load_event_feat((w2i, i2w))

    for filename in filenames:
        with open(os.path.join(CONCRETE_NODEFEAT_JSON_DIR, filename), "r", encoding="utf-8") as f:
            anno_node_feats = ut.filter_dict(lambda k, v: k in announcement2company, json.load(f))

        with open(os.path.join(CONCRETE_GRAPH_DIR, filename.replace("nodefeat", "concrete")), "r",
                  encoding="utf-8") as f:
            id2concrete = json.load(f)
        print("Processing filename {}. There are {} concrete graphs.".format(filename, len(anno_node_feats)))

        anno_node2event = node_event_similarity(anno_node_feats, event_feats)

        anno_com2nodes = {
            announcement_id: hot_stock_company2node(id2concrete[announcement_id])
            if announcement2tag.get(announcement_id) == "熱門股" else {
                com: set(node_mapping.keys()) for com in announcement2company[announcement_id]
            }
            for announcement_id, node_mapping in anno_node2event.items()
        }

        anno_com_node2event = {
            anno_id: {
                com: ut.filter_dict(lambda k, v: v is not None, {
                    node_id: event_select_rule(node_mapping[node_id])
                    for node_id in nodes
                })
                for com, nodes in anno_com2nodes[anno_id].items()
            }
            for anno_id, node_mapping in anno_node2event.items()
        }

        anno_com_node2event = ut.filter_dict(lambda k, v: any([len(vv) > 0 for vv in v.values()]), anno_com_node2event)

        with open(ABSTRACT_NODE_MAPPING_DIR + "/" + filename.replace("nodefeat", "nodemapping"), "w",
                  encoding="utf-8") as f:
            json.dump(anno_com_node2event, f, indent=4)

    return True


"""  OTHERS """


def node_event_mapping_verbose():
    filenames = ut.filter_list(lambda f: f.endswith("json"), os.listdir(CONCRETE_NODEFEAT_JSON_DIR))

    metas = load_efficient_metas()
    announcement2tag = {meta.announcement_id: meta.tag for meta in metas}
    announcement2company = ut.filter_dict(lambda k, v: len(v) > 0,
                                          {meta.announcement_id: company_selector(meta) for meta in metas})

    w2i, i2w = load_vocab_mappings()
    id2event, event_feats = load_event_feat((w2i, i2w))

    for filename in filenames:
        with open(os.path.join(CONCRETE_NODEFEAT_JSON_DIR, filename), "r", encoding="utf-8") as f:
            anno_node_feats = ut.filter_dict(lambda k, v: k in announcement2company, json.load(f))

        with open(os.path.join(CONCRETE_GRAPH_DIR, filename.replace("nodefeat", "concrete")), "r",
                  encoding="utf-8") as f:
            id2concrete = json.load(f)
            tmp_id2node_clause = {anno_id: {
                str(tpl["id"]): " | ".join(tpl["clause"]) for tpl in dct["graph"]["nodes"]}
                for anno_id, dct in id2concrete.items()}
        print("Processing filename {}. There are {} concrete graphs.".format(filename, len(anno_node_feats)))

        anno_node2event = node_event_similarity(anno_node_feats, event_feats)

        anno_com2nodes = {
            announcement_id: hot_stock_company2node(id2concrete[announcement_id])
            if announcement2tag.get(announcement_id) == "熱門股" else {
                com: set(node_mapping.keys()) for com in announcement2company[announcement_id]
            }
            for announcement_id, node_mapping in anno_node2event.items()
        }

        anno_com_node2event = {
            anno_id: {
                com: ut.filter_dict(lambda k, v: v is not None, {
                    node_id: event_select_rule(node_mapping[node_id])
                    for node_id in nodes
                })
                for com, nodes in anno_com2nodes[anno_id].items()
            }
            for anno_id, node_mapping in anno_node2event.items()
        }

        anno2cnt = {anno: len({e
                               for com in anno_com_node2event[anno]
                               for e in anno_com_node2event[anno][com].values()
                               }) for anno in anno_com_node2event}
        for anno in anno2cnt:
            if anno2cnt[anno] < 3: continue
            print("Announcement {}".format(anno))
            for node, events in anno_node2event[anno].items():
                print("{}: {} Feat: {}; \n\t Event:{}".format(
                    node, tmp_id2node_clause[anno][node],
                    ut.flatten([v for k, v in anno_node_feats[anno][node]["feature"].items()]),
                    [(e[0], id2event[e[0]], round(e[1], 2)) for e in events]
                ))

        anno_com_node2event = ut.filter_dict(lambda k, v: any([len(vv) > 0 for vv in v.values()]), anno_com_node2event)

    return True


def remove_conflict(events: list):
    pairs_choose = [("117", "116"), ("119", "118")]
    for p in pairs_choose:
        if p[0] in events and p[1] in events:
            events.remove(p[1])
    return events


def node_event_mapping_headline():
    metas = load_efficient_metas()
    announcement2tag = {meta.announcement_id: meta.tag for meta in metas}
    announcement2company = {meta.announcement_id: company_selector(meta) for meta in metas}
    announcement2headline = {meta.announcement_id: meta.headline for meta in metas}

    w2i, i2w = load_vocab_mappings()
    id2event, event_feats = load_event_feat((w2i, i2w))

    for filename in ut.filter_list(lambda f: f.endswith("json"), os.listdir(HEADLINE_FEATURE_DIR)):
        with open(os.path.join(HEADLINE_FEATURE_DIR, filename), "r", encoding="utf-8") as f:
            headline_feats = ut.filter_dict(lambda k, v: k in announcement2company, json.load(f))
        with open(os.path.join(CONCRETE_NODEFEAT_JSON_DIR, filename.replace("headline", "nodefeat")), "r",
                  encoding="utf-8") as f:
            concrete_node_feats = ut.filter_dict(lambda k, v: k in announcement2company, json.load(f))
        with open(os.path.join(CONCRETE_GRAPH_DIR, filename.replace("headline", "concrete")), "r",
                  encoding="utf-8") as f:
            announcement2concrete = json.load(f)
        print("Processing filename {}. There are {} concrete graphs.".format(filename, len(headline_feats)))

        node_feats = {
            announcement_id: (
                headline_feats[announcement_id]
                if announcement2tag.get(announcement_id) != "熱門股" else concrete_node_feats[announcement_id])
            for announcement_id in concrete_node_feats

        }

        announcement2node2events = node_event_similarity(node_feats, event_feats)

        announcement2com2nodes = {
            announcement_id: hot_stock_company2node(announcement2concrete[announcement_id])
            if announcement2tag.get(announcement_id) == "熱門股" else {
                com: set(node_mapping.keys()) for com in announcement2company[announcement_id]
            }
            for announcement_id, node_mapping in announcement2node2events.items()
        }

        announcement2sign_events = {anno_id: sign_event_rule(announcement2headline[anno_id]) for anno_id in node_feats}

        announcement2com2events = {
            announcement_id: {
                com: remove_conflict(ut.distinct(chain(ut.filter_dict(lambda k, v: v is not None, {
                    node_id: event_select_rule(node2events[node_id])
                    for node_id in nodes
                }).values(), announcement2sign_events[announcement_id])))
                for com, nodes in announcement2com2nodes[announcement_id].items()
            }
            for announcement_id, node2events in announcement2node2events.items()
        }

        announcement2com2events = ut.filter_dict(
            lambda anno_id, com2event: len(com2event) > 0,
            {anno_id: ut.filter_dict(
                lambda com, events: len(events) > 0,
                com2events)
                for anno_id, com2events in announcement2com2events.items()})

        with open(os.path.join(HEADLINE_NODE_MAPPING_DIR, filename.replace("headline", "nodemapping")), "w",
                  encoding="utf-8") as f:
            json.dump(announcement2com2events, f, indent=4)

    return


""" Script  """


def check_headline_mapping():
    metas = load_efficient_metas()
    announcement2tag = {meta.announcement_id: meta.tag for meta in metas}
    announcement2company = {meta.announcement_id: company_selector(meta) for meta in metas}
    announcement2headline = {meta.announcement_id: meta.headline for meta in metas}

    w2i, i2w = load_vocab_mappings()
    id2event, event_feats = load_event_feat((w2i, i2w))

    ff = open("log.txt", "w", encoding="utf-8")

    for filename in ut.filter_list(lambda f: f.endswith("json"), os.listdir(HEADLINE_FEATURE_DIR)):
        with open(os.path.join(HEADLINE_FEATURE_DIR, filename), "r", encoding="utf-8") as f:
            headline_feats = ut.filter_dict(lambda k, v: k in announcement2company, json.load(f))
        with open(os.path.join(CONCRETE_NODEFEAT_JSON_DIR, filename.replace("headline", "nodefeat")), "r",
                  encoding="utf-8") as f:
            concrete_node_feats = ut.filter_dict(lambda k, v: k in announcement2company, json.load(f))
        with open(os.path.join(CONCRETE_GRAPH_DIR, filename.replace("headline", "concrete")), "r",
                  encoding="utf-8") as f:
            announcement2concrete = json.load(f)
        print("Processing filename {}. There are {} concrete graphs.".format(filename, len(headline_feats)))

        node_feats = {
            announcement_id: (
                headline_feats[announcement_id]
                if announcement2tag.get(announcement_id) != "熱門股" else concrete_node_feats[announcement_id])
            for announcement_id in concrete_node_feats

        }

        announcement2node2events = node_event_similarity(node_feats, event_feats)

        announcement2com2nodes = {
            announcement_id: hot_stock_company2node(announcement2concrete[announcement_id])
            if announcement2tag.get(announcement_id) == "熱門股" else {
                com: set(node_mapping.keys()) for com in announcement2company[announcement_id]
            }
            for announcement_id, node_mapping in announcement2node2events.items()
        }

        announcement2sign_events = {anno_id: sign_event_rule(announcement2headline[anno_id]) for anno_id in node_feats}

        announcement2com2events = {
            announcement_id: {
                com: remove_conflict(ut.distinct(chain(ut.filter_dict(lambda k, v: v is not None, {
                    node_id: event_select_rule(node2events[node_id])
                    for node_id in nodes
                }).values(), announcement2sign_events[announcement_id])))
                for com, nodes in announcement2com2nodes[announcement_id].items()
            }
            for announcement_id, node2events in announcement2node2events.items()
        }

        for anno_id, com2events in announcement2com2events.items():
            if any([len(events) > 0 for events in com2events.values()]) or len(com2events) == 0:
                continue
            ff.write("Announcement {} : {} ".format(anno_id, announcement2headline[anno_id]))
            ff.write("\n")
            ff.write("\tFeats : {}; ".format(
                ut.distinct([
                    tuple(feat) for node, feats in node_feats[anno_id].items()
                    for feat in
                    feats["feature"]["general_subject"] + feats["feature"]["verb"] + feats["feature"][
                        "geographical"]
                ])
            ))
            ff.write("\n")

            for com, events in com2events.items():
                ff.write("\tCompany: {}; Cnt: {}; Events : {}".format(
                    com,
                    len(events),
                    events))
                ff.write("\n")
    ff.close()
    return


def check_node_event_mapping(announcement_id):
    filedate = announcement_id.split("-")[0]

    w2i, i2w = load_vocab_mappings()
    id2event, event_feats = load_event_feat((w2i, i2w))
    filename = "nodefeat-" + filedate + ".json"

    with open(os.path.join(CONCRETE_NODEFEAT_JSON_DIR, filename), "r", encoding="utf-8") as f:
        anno_node_feats = ut.filter_dict(lambda k, v: k == announcement_id, json.load(f))

    with open(os.path.join(CONCRETE_GRAPH_DIR, filename.replace("nodefeat", "concrete")), "r",
              encoding="utf-8") as f:
        id2concrete = json.load(f)
        tmp_id2node_clause = {anno_id: {
            str(tpl["id"]): " | ".join(tpl["clause"]) for tpl in dct["graph"]["nodes"]}
            for anno_id, dct in id2concrete.items() if anno_id == announcement_id}

    anno_node2event = node_event_similarity(anno_node_feats, event_feats)

    print("Announcement {}".format(announcement_id))
    for node, events in anno_node2event[announcement_id].items():
        print("{}: {} Feat: {}; \n\t Event:{}".format(
            node, tmp_id2node_clause[announcement_id][node],
            ut.flatten([v for k, v in anno_node_feats[announcement_id][node]["feature"].items()]),
            [(e[0], id2event[e[0]], round(e[1], 2)) for e in events]
        ))


def abstract_event_coverage():
    filenames = ut.filter_list(lambda f: f.endswith("json"), os.listdir(ABSTRACT_NODE_MAPPING_DIR))

    announcement2event_cnt = {}

    for filename in filenames:
        with open(os.path.join(ABSTRACT_NODE_MAPPING_DIR, filename), "r", encoding="utf-8") as f:
            overall_node_mappings = json.load(f)

        id2event_cnt = {
            announcement_id: len({e for node_mappings in com2node_mappings.values() for e in node_mappings.values()})
            for announcement_id, com2node_mappings in overall_node_mappings.items()
        }
        announcement2event_cnt.update(id2event_cnt)

    event_cnt2announcement_cnt = {
        event_cnt: len(announcements)
        for event_cnt, announcements in ut.group_by_key(announcement2event_cnt.items(),
                                                        key=lambda t: t[1], value=lambda t: t[0]).items()
    }
    sum_of_cnt = sum(event_cnt2announcement_cnt.values())
    for event_cnt in sorted(event_cnt2announcement_cnt):
        # print("Event Cnt: {};  Announcement Cnt: {}; Prop: {}".format(
        #     event_cnt, event_cnt2announcement_cnt[event_cnt],
        #     round(event_cnt2announcement_cnt[event_cnt]/sum_of_cnt, 4)))

        print("{}\t{}".format(
            event_cnt2announcement_cnt[event_cnt],
            round(event_cnt2announcement_cnt[event_cnt]/sum_of_cnt, 4)))

    return


if __name__ == '__main__':
    # node_event_mapping_headline()
    # node_event_mapping()
    abstract_event_coverage()
