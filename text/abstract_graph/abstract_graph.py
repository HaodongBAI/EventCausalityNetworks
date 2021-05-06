import json
import os
from collections import defaultdict
from queue import Queue

import networkx as nx

import utils as ut
from config import ABSTRACT_GRAPH_DIR, ABSTRACT_NODE_MAPPING_DIR, CONCRETE_GRAPH_DIR, CONCRETE_NODEFEAT_JSON_DIR
from text.abstract_graph.feature_utils import load_event_feat, load_vocab_mappings


def traverse_event_paths(node_mappings, concrete_graph: nx.MultiDiGraph):
    """
    This used use BFS algorithm to traverse the concrete graphs (which are directed acyclic graphs). The output of
    this function is a dict that contains shortest path length for all event pairs.

    Node_message is used to memory the shortest length between previous events to the current node.
    node_message := { node_id : { event_id : shortest_length }}

    :param node_mappings: a dict where key is node id and value is event id
    :param concrete_graph: a nx.MultiDiGraph object
    :return event_path: a dict where key is a ordered pair of events and value is shortest path length in concrete_graph
    """
    node_message = defaultdict(dict)
    q = Queue()
    for node in node_mappings:
        node_message[node] = {node_mappings[node]: 0}

    visited = set()
    for node in filter(lambda n: concrete_graph.in_degree[n] == 0, concrete_graph.nodes):
        q.put(node)

    while q.qsize() != 0:
        node = q.get()
        visited.add(node)

        for n in concrete_graph.succ[node]:
            if not all([np in visited for np in concrete_graph.pred[n]]):
                continue

            # All pred nodes of n have been visited.
            q.put(n)
            node_message[n] = {
                event: min(path_lengths) + 1
                for event, path_lengths in ut.group_by_key(
                [(e, l) for np in concrete_graph.pred[n] for e, l in node_message[np].items()],
                key=lambda tpl: tpl[0],
                value=lambda tpl: tpl[1]
            ).items()}

    event_path = {
        (ns, node_mappings[ne]): node_message[ne][ns]
        for ne in node_mappings for ns in node_message[ne] if ns != node_mappings[ne]
    }

    return event_path


def build_abstract_graph():
    """
    This function builds abstract_graph from all node mappings and all concrete graphs. Abstract graph is a
    Multidigraph object which allows two direction edge between two nodes. Nodes in abstract_graph have
    two attributes "event" and "cnt". "event" is event description. "cnt" is occurance in all concrete_graphs of this
    event. Edges in abstract_graph have one attribute "weight".

        weight := sum( 1 / path_length )


    :return:
    """
    filenames = ut.filter_list(lambda f: f.endswith("json"), os.listdir(ABSTRACT_NODE_MAPPING_DIR))

    abstract_graph = nx.MultiDiGraph()
    w2i, i2w = load_vocab_mappings()
    id2event, event_feats = load_event_feat((w2i, i2w))
    abstract_graph.add_nodes_from([
        (event, {"event": event_des, "cnt": 0})
        for event, event_des in id2event.items()
    ])

    for filename in filenames:
        with open(os.path.join(ABSTRACT_NODE_MAPPING_DIR, filename), "r", encoding="utf-8") as f:
            overall_node_mappings = json.load(f)
        with open(os.path.join(CONCRETE_GRAPH_DIR, filename.replace("nodemapping", "concrete")), "r",
                  encoding="utf-8") as f:
            id2concrete = json.load(f)

        print("Processing filename {}. There are {} concrete graphs.".format(filename, len(overall_node_mappings)))

        for announcement_id, com2node_mappings in overall_node_mappings.items():
            concrete_graph = nx.node_link_graph(id2concrete[announcement_id]["graph"])
            node_mappings = {int(node): event for com, node_mappings in com2node_mappings.items()
                             for node, event in node_mappings.items()}
            edge_paths = traverse_event_paths(node_mappings, concrete_graph)

            for node, event in node_mappings.items():
                abstract_graph.nodes[event]["cnt"] += 1

            for (edge_u, edge_v), path_length in edge_paths.items():
                if abstract_graph.has_edge(edge_u, edge_v):
                    abstract_graph.edges[edge_u, edge_v, 0]["weight"] += 1/path_length
                else:
                    abstract_graph.add_edge(edge_u, edge_v, weight=1/path_length)

    for edge_u, edge_v, k in list(abstract_graph.edges):
        if abstract_graph.edges[edge_u, edge_v, 0]["weight"] >= 5: continue
        abstract_graph.remove_edge(edge_u, edge_v, k)

    with open(os.path.join(ABSTRACT_GRAPH_DIR, "abstract_graph.json"), "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(abstract_graph), f, indent=4, ensure_ascii=False)

    return True

""" script """
def check_abstract_graph():
    with open(os.path.join(ABSTRACT_GRAPH_DIR, "abstract_graph.json"), "r", encoding="utf-8") as f:
        abstract_graph = json.load(f)

    id2event = {node["id"]: node["event"] for node in abstract_graph["nodes"]}
    for node in abstract_graph["nodes"]:
        print(node["event"], node["cnt"])

    for link in sorted(abstract_graph["links"], key=lambda dct: dct["weight"], reverse=True):
        print(id2event[link["source"]], id2event[link["target"]], link["weight"])
    return


def check_event_nodes(event=None):
    filenames = ut.filter_list(lambda f: f.endswith("json"), os.listdir(ABSTRACT_NODE_MAPPING_DIR))

    for filename in filenames:

        with open(os.path.join(ABSTRACT_NODE_MAPPING_DIR, filename), "r", encoding="utf-8") as f:
            announcement_com2node_mappings = json.load(f)
        announcement2node_mappings = {anno: {
            node: event for com, d in dct.items() for node, event in d.items()}
            for anno, dct in announcement_com2node_mappings.items()}

        with open(os.path.join(CONCRETE_GRAPH_DIR, filename.replace("nodemapping", "concrete")), "r",
                  encoding="utf-8") as f:
            id2concrete = json.load(f)

        with open(os.path.join(CONCRETE_NODEFEAT_JSON_DIR, filename.replace("nodemapping", "nodefeat")), "r",
                  encoding="utf-8") as f:
            anno_node_feats = json.load(f)

        for announcement_id in announcement2node_mappings:
            node2clause = {str(dct["id"]): " | ".join(dct["clause"]) for dct in
                           id2concrete[announcement_id]["graph"]["nodes"]}
            for node, e in announcement2node_mappings[announcement_id].items():
                if e != event: continue
                print(node2clause[node])
                print(
                    "\t{}".format(
                        [feat for k in ["general_subject", "verb", "geographical"]
                         for feat in anno_node_feats[announcement_id][node]["feature"][k]]))

    return


def check_transfer_distribution(event):
    with open(os.path.join(ABSTRACT_GRAPH_DIR, "abstract_graph.json"), "r", encoding="utf-8") as f:
        abstract_graph = json.load(f)
    id2event = {node["id"]: node["event"] for node in abstract_graph["nodes"]}

    transfer = defaultdict(dict)
    for link in sorted(abstract_graph["links"], key=lambda dct: dct["weight"], reverse=True):
        transfer[link["source"]][link["target"]] = link["weight"]

    return


def check_abstract_graph_edge():
    with open(os.path.join(ABSTRACT_GRAPH_DIR, "abstract_graph_normalized.json"), "r", encoding="utf-8") as f:
        abstract_graph = json.load(f)
    id2event = {node["id"]: node["event"] for node in abstract_graph["nodes"]}
    id2cnt = {node["id"]: node["cnt"] for node in abstract_graph["nodes"]}
    # for node in abstract_graph["nodes"]:
    #     print(node["event"], node["cnt"])
    #
    # for link in sorted(abstract_graph["links"], key=lambda dct: dct["weight"], reverse=True):
    #     print(id2event[link["source"]], id2event[link["target"]], link["weight"])

    adj_weights = {(link["source"], link["target"]): link["normalized_weight"] for link in abstract_graph["links"]}
    adj_nodes = ut.group_by_key(adj_weights, key=lambda t: t[0], value=lambda t: t[1])
    nested_adj_weights = {s: {e: adj_weights[(s, e)] for e in adj_nodes[s]} for s in adj_nodes}
    for id in sorted(id2cnt, key=lambda t: id2cnt[t], reverse=True):
        print("{id}\t{event}\t{cnt}\t{edges}".format(id=id, event=id2event[id], cnt=id2cnt[id], edges="\t".join([
            "{e}\t{w}".format(e=id2event[e], w=round(weight, 4)) for e, weight in sorted(
                nested_adj_weights[id].items(), key=lambda t: t[1], reverse=True
            )[:5]
        ])))

    return


if __name__ == '__main__':
    check_abstract_graph_edge()
