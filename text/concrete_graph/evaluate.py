import networkx as nx
import utils as ut
import re
from functools import reduce
from queue import Queue
import json
from config import CONCRETE_GRAPH_DIR
import traceback


def has_path(g: nx.DiGraph, n1, n2):
    # 判断节点间是否有路径,其实可以使用networkx中的函数
    q = Queue()
    q.put(n1)
    visited = set()
    while not q.empty():
        n = q.get()
        visited.add(n)
        for s in g.succ[n]:
            if s == n2: return True
            if s in visited: continue
            q.put(s)
    return False


def evaluate_graph(clause_graph: nx.DiGraph, golden):
    # Statis for nodes
    node_mapping = {}
    node2golden = {n: {w for w in d["clause"]} for n, d in golden.nodes(data=True)}
    node2clause = {n: {w for c in d["clause"] for w in c} for n, d in clause_graph.nodes(data=True)}
    for node, golden_clause in node2golden.items():
        node2intersect = {n: len(v.intersection(golden_clause))/min(len(golden_clause), len(v))
                          for n, v in node2clause.items()}
        exist_node = reduce(lambda s, t: s if s[1] > t[1] else t, node2intersect.items())
        node_mapping[node] = exist_node[0]
    node_found = len({v for v in node_mapping.values()})
    node_not_found = len(node_mapping) - node_found

    # Statis for edges
    edge_found = 0
    edge_path_found = 0
    edge_not_found = 0
    for u, v in golden.edges():
        eu, ev = node_mapping.get(u), node_mapping.get(v)
        if eu is None or ev is None: continue
        if clause_graph.has_edge(eu, ev):
            edge_found += 1
        elif has_path(clause_graph, eu, ev):
            edge_path_found += 1
        else:
            edge_not_found += 1

    return {
        "node_found"     : node_found,
        "node_not_found" : node_not_found,
        "edge_found"     : edge_found,
        "edge_path_found": edge_path_found,
        "edge_not_found" : edge_not_found,
    }


def parse_golden_graph(lines):
    # 解析标准答案
    lines = ut.filter_list(lambda l: len(l.strip()) > 0, lines)
    id_line, node_lines, edge_line = lines[0], lines[1:-1], lines[-1]
    id = re.findall(r"Announcement:\s([\d\-]+)", id_line)[0]
    edges = [(int(res[0]), int(res[1])) for res in re.findall(r"\((\d+),(\d+)\)", edge_line)]
    nodes = [(int(res[0]), res[1]) for l in node_lines for res in [l.strip().split()]]

    g = nx.DiGraph()
    g.add_nodes_from([
        (n[0], {"clause": n[1]})
        for n in nodes
    ])
    g.add_edges_from([(u, v) for u, v in edges])
    return id, g


def reindex_golden_graph(ifilename, ofilename):
    # 将人工编写的标准答案进行重整
    with open(ifilename, "r", encoding="utf-8") as f:
        golden_lines = list(f.readlines())
    starts, ends = ut.split_list(golden_lines, lambda _, l: len(l.strip()) == 0)

    output = []
    for s, e in zip(starts, ends):
        lines = golden_lines[s:e]
        lines = ut.filter_list(lambda l: len(l.strip()) > 0, lines)
        id_line, node_lines, edge_line = lines[0], lines[1:-1], lines[-1]
        edges = [(int(res[0]), int(res[1])) for res in re.findall(r"\((\d+),(\d+)\)", edge_line)]
        nodes = [(int(res[0]), res[1]) for l in node_lines for res in [l.strip().split()]]
        node_mapping = {n[0]: i for i, n in enumerate(nodes)}
        try:
            output.extend([
                id_line,
                *[
                    "{}\t{}".format(node_mapping[node[0]], node[1]) + "\n"
                    for node in nodes
                ],
                ",".join([
                    "({},{})".format(u, v)
                    for u, v in sorted([(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in edges])]
                ) + "\n",
                "\n"
            ])
        except Exception as e:
            print("An error occurs when processing {}".format(id_line))
            traceback.print_exc()

    with open(ofilename, "w", encoding="utf-8") as f:
        f.writelines(output)
    return True


def announcement_id_in_golden():
    # 遍历得到所有标准答案中的announcement-id
    gfilename = CONCRETE_GRAPH_DIR + "/concrete-golden.txt"
    with open(gfilename, "r", encoding="utf-8") as f:
        golden_lines = list(f.readlines())
    starts, ends = ut.split_list(golden_lines, lambda _, l: len(l.strip()) == 0)
    id_lines = [golden_lines[s] for s in starts]
    ids = [re.findall(r"Announcement:\s([\d\-]+)", id_line)[0] for id_line in id_lines]
    for id in ids:
        print("\"{}\",".format(id))
    return True


""" 入口函数 """


def evaluate():
    concrete_graphs = {}
    jfilenames = ["/concrete-20180102.json", "/concrete-20180702.json", "/concrete-20180402.json", ]

    for jfilename in jfilenames:
        with open(CONCRETE_GRAPH_DIR + jfilename, "r", encoding="utf-8") as f:
            concrete_graphs.update(json.load(f))

    gfilename = CONCRETE_GRAPH_DIR + "/concrete-golden.txt"
    with open(gfilename, "r", encoding="utf-8") as f:
        golden_lines = list(f.readlines())
    starts, ends = ut.split_list(golden_lines, lambda _, l: len(l.strip()) == 0)

    np = []
    nr = []
    ep = []
    er = []
    epp = []
    epr = []

    for i, (s, e) in enumerate(zip(starts, ends)):
        id, golden = parse_golden_graph(golden_lines[s:e])
        if id not in concrete_graphs: continue
        concrete = nx.node_link_graph(concrete_graphs[id]["graph"])
        statis = concrete_graphs[id]["statis"]
        pr_statis = evaluate_graph(concrete, golden)

        node_precision = pr_statis["node_found"]/statis["node_after_fuse"]
        node_recall = pr_statis["node_found"]/(pr_statis["node_found"] + pr_statis["node_not_found"])

        edge_precision = pr_statis["edge_found"]/statis["edge_after_fuse"]
        edge_recall = pr_statis["edge_found"]/(
            pr_statis["edge_found"] + pr_statis["edge_path_found"] + pr_statis["edge_not_found"])

        edge_path_precision = (pr_statis["edge_found"] + pr_statis["edge_path_found"])/statis["edge_after_fuse"]
        edge_path_recall = (pr_statis["edge_found"] + pr_statis["edge_path_found"])/(
            pr_statis["edge_found"] + pr_statis["edge_path_found"] + pr_statis["edge_not_found"])

        print("{}. Announcement {}: {}.".format(i, id, "; ".join([
            ("\033[31m{}\033[0m" if node_precision <= 0.7 else "{}").format("NP %.2f"%node_precision),
            ("\033[31m{}\033[0m" if node_recall <= 0.7 else "{}").format("NR %.2f"%node_recall),
            ("\033[31m{}\033[0m" if edge_precision <= 0.3 else "{}").format("EP %.2f"%edge_precision),
            ("\033[31m{}\033[0m" if edge_recall <= 0.5 else "{}").format("ER %.2f"%edge_recall),
            ("\033[31m{}\033[0m" if edge_path_precision <= 0.4 else "{}").format("EpP %.2f"%edge_path_precision),
            ("\033[31m{}\033[0m" if edge_path_recall <= 0.6 else "{}").format("EpR %.2f"%edge_path_recall),
        ])))

        np.append(node_precision)
        nr.append(node_recall)
        ep.append(edge_precision)
        er.append(edge_recall)
        epp.append(edge_path_precision)
        epr.append(edge_path_recall)

    print("Overall: {}.".format(
        "; ".join([
            "Total %i"%len(np),
            "NP %.4f"%(sum(np)/len(np)),
            "NR %.4f"%(sum(nr)/len(nr)),
            "EP %.4f"%(sum(ep)/len(ep)),
            "ER %.4f"%(sum(er)/len(er)),
            "EpP %.4f"%(sum(epp)/len(epp)),
            "EpR %.4f"%(sum(epr)/len(epr)),
        ])
    ))

    return


if __name__ == '__main__':
    # reindex_golden_graph(CONCRETE_GRAPH_DIR+"/concrete-golden.txt", CONCRETE_GRAPH_DIR+"/text.txt")
    # announcement_id_in_golden()
    evaluate()
    # reformat_concrete_graph(CONCRETE_GRAPH_DIR+"/concrete-20180402.json",
    # CONCRETE_GRAPH_DIR+"/concrete-golden-0402.txt")
