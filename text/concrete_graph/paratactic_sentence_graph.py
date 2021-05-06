import queue
import re
from functools import reduce
from itertools import permutations, product

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

import utils as ut
from concrete_graph.causality_extractor import default_causality_extractor
from concrete_graph.compound_extractor import default_compound_extractor
from concrete_graph.conf import COMMON_SBV_RULE_CONTROL, FILTER_SEGMENT_VERBOSE, MERGE_RULE_VERBOSE, \
    PARSED_RELATION_VERBOSE
from config import CONCRETE_VOCAB_DIR

matplotlib.rcParams['font.sans-serif'] = ['SimHei']

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['figure.figsize'] = (15, 15)


def plot_graph(g, node_label, edge_label=None, node_color=None, filename=None):
    # plot segment tree
    # pos = nx.spiral_layout(g)

    def topology_position(g):
        pos = {}
        node_queue = queue.Queue()
        for node in filter(lambda n: g.in_degree[n] == 0, g):
            node_queue.put(node)

        s = node_queue.qsize()
        level = 0
        visited = set()
        while not node_queue.empty():
            node = node_queue.get()
            visited.add(node)
            s -= 1
            pos[node] = (s, level+(0.6 * s / 5))
            for s_node in g.succ[node]:
                if s_node in visited: continue
                node_queue.put(s_node)
            if s == 0:
                level += 1
                s = node_queue.qsize()

        return pos

    pos = topology_position(g)
    # pos = nx.spiral_layout(g)
    nx.draw_networkx(g, pos, with_labels=True, node_shape="o", node_size=500, node_color=node_color,
                     labels=node_label)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_label, label_pos=0.3)
    if filename:
        plt.savefig(filename, format="PNG")
        plt.close()
    else:
        plt.show()
    return


"""     Graph Manipulation      """


def merge_nodes(g: nx.DiGraph, nodes, merger):
    if len(nodes) < 2: return g
    new_node = max(filter(lambda n: isinstance(n, int), g.nodes))+1
    g.add_node(new_node)

    g = merger(g, new_node, nodes)

    g.add_edges_from([
        (pred, new_node, data)
        for seg in nodes for pred, data in g.pred[seg].items()])
    g.add_edges_from([
        (new_node, succ, data)
        for seg in nodes for succ, data in g.succ[seg].items()])

    g.remove_nodes_from(nodes)
    return g


def split_node(g: nx.DiGraph, node, splitter, k=2, mode="horizontal"):
    # mode = "vertical|horizontal"

    new_nodes = [0] * k
    new_nodes[0] = max(filter(lambda n: isinstance(n, int), g.nodes))+1
    for i in range(1, k):
        new_nodes[i] = new_nodes[i-1]+1

    g.add_nodes_from(new_nodes)

    g = splitter(g, new_nodes, node)
    if mode == "vertical":
        g.add_edges_from([
            (pred, new_node, data) for pred, data in g.pred[node].items() for new_node in new_nodes])
        g.add_edges_from([
            (new_node, succ, data) for succ, data in g.succ[node].items() for new_node in new_nodes])
    else:
        g.add_edges_from([
            (pred, new_nodes[0], data) for pred, data in g.pred[node].items()])
        g.add_edges_from([
            (new_nodes[-1], succ, data) for succ, data in g.succ[node].items()])
    g.remove_node(node)

    return g


def parallelize_node(g: nx.DiGraph, node_pred, node_succ):
    # 一般来讲 node_pred ==  node_succ, 如果遇到一些已经形成的因果子句就会顺延succ
    preds = g.pred[node_pred].keys()
    succs = g.succ[node_succ].keys()
    g.add_edges_from(
        ut.flatten([[(pred, succ, data) for succ in succs]
                    for pred, data in g.pred[node_pred].items()])
    )
    g.remove_edges_from([(pred, node_pred) for pred in preds])
    return g


def remove_node(g: nx.DiGraph, node):
    g.add_edges_from([
        (pred, succ, ut.dict_union(p_data, s_data))
        for (pred, p_data), (succ, s_data) in product(g.pred[node].items(), g.succ[node].items())
    ])
    g.remove_node(node)
    return g


"""     Concrete CE-Graph Creation      """

with open(CONCRETE_VOCAB_DIR+"/psych_verbs", "r", encoding="UTF-8") as f:
    psych_verbs = {l.strip() for l in f.readlines()}

with open(CONCRETE_VOCAB_DIR+"/coord_conjs", "r", encoding="UTF-8") as f:
    coord_conjs = {l.strip() for l in f.readlines()}

__SBV_structure = lambda range, arcs: all(
    [r.relation in {"SBV", "ADV"} for r in arcs[range[0]:range[1]] if not (range[0] <= r.head-1 < range[1])])

"""MERGE 最核心规则"""
def __merge_rule(g, nodes, words, postags, arcs, roles):
    """
    最核心的逻辑,输入是一些节点id,g中的属性将这些id映射为range

    """
    role_ranges = [(min([arc.range.start for arc in role.arguments]),
                    max([arc.range.end for arc in role.arguments])) for role in roles]

    # 如果node开始为心理动词,那么就和前一个割裂开
    def psych_verb_rule(g: nx.DiGraph, nodes):

        for n2 in nodes[1:]:
            n2s, n2e = g.nodes[n2]["range"]
            if words[n2s] in psych_verbs or (
                        words[n2e-1] in psych_verbs and __SBV_structure((n2s, n2e-1), arcs)):
                # If one segment is started with Psych-Verb or SBV ++ Psych-Verb structure,
                # then this segment is separated from the previous one.
                return False
        else:
            return True

    # 如果node之间为分号,就分开
    def semicolon_rule(g: nx.DiGraph, nodes):
        ranges = ut.filter_list(lambda r: 0 <= r[1] < len(words), [g.nodes[n]["range"] for n in nodes])
        return all([words[r[1]] != "；" for r in ranges])

    # 如果node的range并不是连通的,那么不合并
    def continuation_rule(g: nx.DiGraph, nodes):
        ranges = [g.nodes[n]["range"] for n in nodes]
        return all([r1[1] == r2[0]-1 for r1, r2 in zip(ranges[:-1], ranges[1:])])

    # 如果node第一个词为连词,那么分开
    def extra_sentence_conj_rule(g: nx.DiGraph, nodes):
        return all([words[g.nodes[n]["range"][0]] not in coord_conjs for n in nodes[1:]])

    # 如果两个节点之间的关系不是并列关系,那么合并
    def non_coo_rule(g: nx.DiGraph, nodes):
        n1, n2 = nodes[0], nodes[-1]
        if g.has_edge(n1, n2):
            return g[n1][n2]["relation"] not in {"COO", "HED"}
        elif g.has_edge(n2, n1):
            return g[n2][n1]["relation"] not in {"COO", "HED"}
        else:
            return False

    # 如果两个node存在同一个语义角色,那么合并
    def common_semantic_role_rule(g: nx.DiGraph, nodes):
        # overlap role and segments
        n1, n2 = nodes[0], nodes[-1]
        n1s, n1e = g.nodes[n1]["range"]
        n2s, n2e = g.nodes[n2]["range"]
        return any([n1s <= s < n1e and n2s <= e < n2e for (s, e) in role_ranges])

    # 如果节点具有共同的sbv,那么合并
    def common_sbv_rule(g: nx.DiGraph, nodes):

        n0, n_coo = nodes[0], nodes[1:]
        if any([n_p not in nodes for n in n_coo for n_p in g.pred[n]]):
            # if any pred of nodes in n_coo is not in `nodes`, then split nodes
            return False
        starts = [g.nodes[n]["range"][0] for n in nodes]
        ends = [g.nodes[n]["range"][1] for n in nodes]
        sub_arcs = [{arcs[i].relation for i in range(starts[i], ends[i])}
                    for i in range(len(nodes))]

        # If the first segment has SBV and the second segment has only VOB/CMP but no SBV (we called it Common-SBV),
        # then we need to merge two segments if this is parsed correctly (There is no real SBV in the second segment).
        if "SBV" in sub_arcs[0] and all([
                    "SBV" not in sub_arc and ("VOB" in sub_arc or "CMP" in sub_arc)
            for sub_arc in sub_arcs[1:]
        ]):
            verb_relative_indices = [ut.find(range(s, e),
                                             lambda _, idx: arcs[idx].head-1 >= e or arcs[idx].head-1 < s)
                                     for s, e in zip(starts[1:], ends[1:])]

            # coo-verb is close to comma
            return all([v < 2 for v in verb_relative_indices])
        else:
            return False

    # 如果可以从中挖掘出因果关系,那么合并
    def causality_rule(g: nx.DiGraph, nodes):
        if len(nodes) > 2:
            flag = __merge_rule(g, nodes[1:], words, postags, arcs, roles)
        else:
            flag = True
        start = g.nodes[nodes[0]]["range"][0]
        end = g.nodes[nodes[1]]["range"][1]
        return flag and default_causality_extractor.has_extraction(words[start:end], postags[start:end])

    # 如果从中可以挖掘出并列关系,那么合并
    def compound_rule(g: nx.DiGraph, nodes):
        start = g.nodes[nodes[0]]["range"][0]
        end = g.nodes[nodes[1]]["range"][1]
        return default_compound_extractor.has_extraction(words[start:end], postags[start:end])

    # 如果首词为包括等,那么合并
    def contain_rule(g: nx.DiGraph, nodes):
        ranges = [g.nodes[n]["range"] for n in nodes]
        s = ranges[1][0]
        if len(ranges) == 2:
            return words[s] in {"包括", "包含"} or (postags[s] in {"p", "c"} and words[s+1] in {"包括", "包含"})
        else:
            flag = __merge_rule(g, nodes[1:], words, postags, arcs, roles)
            return flag and words[s] in {"包括", "包含"} or (postags[s] in {"p", "c"} and words[s+1] in {"包括", "包含"})

    if MERGE_RULE_VERBOSE:
        # 测试用,用于输出是否需要调整规则
        p = psych_verb_rule(g, nodes)
        s = semicolon_rule(g, nodes)
        ct = continuation_rule(g, nodes)
        ex = extra_sentence_conj_rule(g, nodes)
        c = non_coo_rule(g, nodes)
        cs = common_semantic_role_rule(g, nodes)
        sbv = common_sbv_rule(g, nodes)
        ca = causality_rule(g, nodes)
        i = compound_rule(g, nodes)
        cn = contain_rule(g, nodes)
        print("Merge nodes : {}".format(
            " | ".join(["".join(words[r[0]:r[1]]) for n in nodes for r in [g.nodes[n]["range"], ]])
        ))
        print("Meet rules: Ultimate:{}; Psych:{}; Semi:{}; Cont:{}; Extra:{}; Coo:{}; Semantic:{}; SBV:{}; "
              "Causality:{}; Intra:{}; Contain:{};".format(
            p and s and ct and ex and (c or cs or (sbv if COMMON_SBV_RULE_CONTROL else False) or ca or i or cn), p, s,
            ct, ex, c, cs, sbv, ca, i, cn
        ))
        return p and s and ct and ex and (c or cs or (sbv if COMMON_SBV_RULE_CONTROL else False) or ca or i or cn)

    return psych_verb_rule(g, nodes) and semicolon_rule(g, nodes) and continuation_rule(g, nodes) and \
           extra_sentence_conj_rule(g, nodes) and (
               non_coo_rule(g, nodes) or common_semantic_role_rule(g, nodes) or
               (common_sbv_rule(g, nodes) if COMMON_SBV_RULE_CONTROL else False) or
               compound_rule(g, nodes) or contain_rule(g, nodes) or causality_rule(g, nodes)
           )


def __compound_rule(g, node, words, postags, arcs, roles):
    s = g.nodes[node]["range"][0][0]
    return len(g.succ[node]) > 0 and words[s] in coord_conjs


def __filter_rule(s, e, words, postags, arcs, roles):
    if e == s: return True
    # SBV + psych_verb
    sbv_v = e > s+1 and words[e-1] in psych_verbs and __SBV_structure((s, e-1), arcs)
    # SBV + psych_verb + psych_verb
    sbv_v_v = e > s+2 and words[e-1] in psych_verbs and words[e-2] in psych_verbs and __SBV_structure(
        (s, e-2), arcs)
    # non noun words
    single = e-s < 3 and all([p not in {"v"} and not p.startswith("n") for p in postags[s:e]])
    # 据...psych-verb
    accord = e-s > 2 and words[s] in {"据", "根据"} and words[e-1] in psych_verbs
    # 展望2018年; 进入第三季
    pos_pt = re.match(r"^v_nt$", "_".join(postags[s:e]))
    # 并看好
    conj_v = (e-s == 2) and words[e-1] in psych_verbs and postags[s] in {"c", "p"}
    if FILTER_SEGMENT_VERBOSE:
        print("Filter segment : {}, {}".format("|".join(words[s:e]), "|".join([a.relation for a in arcs[s:e]])))
        print("Meet rules: SBV-V:{}; SBV-VV:{}; Single:{}; Accord:{}; V-NT:{}; Conj-v".format(
            sbv_v, sbv_v_v, single, accord, pos_pt, conj_v
        ))
    return sbv_v or sbv_v_v or single or accord or pos_pt or conj_v


def __cut_link_rule(g, n1, n2, words, postags, arcs, roles):
    s2 = g.nodes[n2]["range"][0]
    return words[s2] in {"至于"} or words[s2-1] in {"；"}


def build_segment_tree(words, postags, arcs, roles):
    """
    This function is used to create segment tree. One segment is defined as a word sequence that ends with a comma.
    Segment tree is represented as a networkx.DiGraph object. ROOT node is denoted as "r" and other segment nodes are
    denoted as numbers.

    Node attributes: "range"(tuple of strat and end index of word sub sequence); "words"(word sub sequence).
    Edge attributes: "relation"(relation arc which connect this two segments); "word"(word where relation from).

    :param words: word sequence.
    :param arcs: are relation sequence, where head-1 correspond to index in word sequence.
    :param roles: [role.index, role.arguments]; arguments: [arg.name, arg.range.start, arg.range.end]
    :return:
        segment_graph: networkx.DiGraph object to represent a segment tree.
    """

    # create segment tree
    segment_graph = nx.DiGraph()

    comma = {"，", "；", "：", "，"}
    start, end = ut.split_list(words, lambda _, w: w in comma)
    # create nodes
    for i, (s, e) in enumerate(zip(start, end)):
        if __filter_rule(s, e, words, postags, arcs, roles): continue
        segment_graph.add_node(i, range=(s, e))
    segment_graph.add_node("r", words="ROOT")

    # create edges
    for node_i, node_j in permutations(filter(lambda n: n != "r", segment_graph.nodes), 2):
        s_i, e_i = segment_graph.nodes[node_i]["range"]
        s_j, e_j = segment_graph.nodes[node_j]["range"]
        for word, arc in zip(words[s_i:e_i], arcs[s_i:e_i]):
            head, relation = arc.head, arc.relation
            if arc.relation == "WP": continue
            if head == 0:  # related to "ROOT" node
                segment_graph.add_edge("r", node_i, relation=relation, word=word)
            if s_j <= head-1 < e_j:  # related to other segment node
                segment_graph.add_edge(node_j, node_i, relation=relation, word=word)

    if PARSED_RELATION_VERBOSE:
        plot_graph(segment_graph,
                   node_label={n: "".join(words[d["range"][0]:d["range"][1]]) if "range" in d else "ROOT"
                               for n, d in segment_graph.nodes(data=True)},
                   edge_label={(u, v): d["relation"] for u, v, d in segment_graph.edges(data=True)})

    return segment_graph


def merge_segments(segment_graph: nx.DiGraph, words, postags, arcs, roles):
    """
    This function is used to merge some segments in order to extract causality as easily as possible.
    The extraction will be carried out only on each segment individually.
    So there should not be causality across multiple segments.

    All of this are guaranteed by "MERGE_RULE".

    :param segment_graph: networkx.DiGraph object to represent a segment tree.
    :param words: word list
    :param postags: pos_tag list
    :param arcs: [arc.head, arc.relation] list
    :param roles: [role.index, role.arguments]; arguments: [arg.name, arg.range.start, arg.range.end]
    :return:
        clause_graph: networkx.DiGraph object to represent a clause tree.
    """
    # use union-find set to find segments to merge
    union_find = ut.UnionFind(segment_graph.nodes)

    nodes = sorted(filter(lambda n: n != "r", segment_graph), key=lambda n: segment_graph.nodes[n]["range"][0])
    for i in range(len(nodes)-2):
        if __merge_rule(segment_graph, nodes[i:(i+3)], words, postags, arcs, roles):
            union_find.union(nodes[i], nodes[i+1])
            union_find.union(nodes[i], nodes[i+2])

    for i in range(len(nodes)-1):
        if not (union_find.self_cluster(nodes[i]) and union_find.self_cluster(nodes[i+1])): continue
        if __merge_rule(segment_graph, nodes[i:(i+2)], words, postags, arcs, roles):
            union_find.union(nodes[i], nodes[i+1])

    # call-back function merger
    def merger(g, new_node, nodes):
        ranges = [g.nodes[node]["range"] for node in sorted(nodes)]
        if any([r_0[1] != r_1[0]-1 for r_0, r_1 in zip(ranges[:-1], ranges[1:])]):
            # nodes that need to merge are not continuous; merge-rule is not correct for all parsed sentence
            raise ValueError("Non-continuous parsed segments to be merged")
        g.nodes[new_node]["range"] = reduce(lambda t, s: (min(t[0], s[0]), max(t[1], s[1])), ranges)
        return g

    # merge nodes in graph
    segment_graph = reduce(
        lambda sg, nodes: merge_nodes(sg, nodes, merger),
        filter(lambda nodes: "r" not in nodes,
               union_find.root2ele.values()), segment_graph)

    clause_graph = nx.DiGraph()
    node_datas = sorted(filter(lambda tpl: tpl[0] != "r", segment_graph.nodes(data=True)),
                        key=lambda tpl: tpl[1]["range"][0])
    nodes = [tpl[0] for tpl in node_datas]
    clause_graph.add_nodes_from(node_datas)

    clause_graph.add_edges_from(
        [(u, v) for u, v in zip(nodes[:-1], nodes[1:])
         if not __cut_link_rule(clause_graph, u, v, words, postags, arcs, roles)]
    )

    for clause_node in clause_graph:
        clause_graph.nodes[clause_node]["range"] = [clause_graph.nodes[clause_node]["range"], ]

    return clause_graph


def extract_clauses(clause_graph: nx.DiGraph, words, postags, arcs, roles):
    """
    This function extract causality using Extractor from each one segment node.
    Firstly, we copy segment graph as clause graph. Then, we split segment node if there is any extraction result.
    Note that we left all extraction assumptions in Extractor and we only get one extraction here.
    Node split is simply finished by connecting all preds to cause-node and all succs to effect-node.

    :param clause_graph: networkx.DiGraph object to represent a clause tree.
    :param words: word list
    :param postags: pos_tag list
    :param arcs: [arc.head, arc.relation] list
    :param roles: [role.index, role.arguments]; arguments: [arg.name, arg.range.start, arg.range.end]
    :return:
        clause_graph: networkx.DiGraph object to represent a clause tree. Node has attribute range which is list of
        index tuple
    """

    def cause_effect_splitter_wrapper(datas):
        """
        ranges = [(5,13),]
        data = [{"cause":[(0,1), (5,8)], "effect":[2,4]}, ]

        cause_range = [(5,6), (10,13)]
        effect_range = [(7,9),]
        :param data:
        :return:
        """

        def splitter(g: nx.DiGraph, new_nodes, node):
            left, right = new_nodes[0], new_nodes[1]
            s, e = g.nodes[node]["range"][0]
            cause_range = [(s+c_range[0], s+c_range[1]) for c_range in datas[0]["cause"]]
            effect_range = [(s+e_range[0], s+e_range[1]) for e_range in datas[0]["effect"]]
            g.nodes[left]["range"] = cause_range
            g.nodes[right]["range"] = effect_range
            g.add_edge(left, right, tag=datas[0]["tag"])
            return g

        return splitter

    def compound_splitter_wrapper(datas):
        """
        ranges = [(0,3), (5,8)]
        data = [{},{"left":(0,1), "right":(2,3)}]

        data = [{}, {"compound": [(0,1), (2,3)] } ]

        node_1_range = [(0,3), (5,6)]
        node_2_range = [(0,3), (7,8)]
        :param data:
        :return:
        """

        def splitter(g, new_nodes, node):
            ranges = g.nodes[node]["range"]
            for i in range(len(new_nodes)):
                node_i_range = [(s, e) if len(data) == 0 else (s+data["compound"][i][0], s+data["compound"][i][1])
                                for (s, e), data in zip(ranges, datas)]
                g.nodes[new_nodes[i]]["range"] = node_i_range

            return g

        return splitter

    causality_extractor = default_causality_extractor
    compound_extractor = default_compound_extractor

    for clause_node in list(clause_graph.nodes):
        ranges = clause_graph.nodes[clause_node]["range"]
        datas = [causality_extractor.extract_pyltp(words[s:e], postags[s:e]) for s, e in ranges]
        if all([len(data) == 0 for data in datas]): continue
        clause_graph = split_node(clause_graph, clause_node, cause_effect_splitter_wrapper(datas),
                                  k=2, mode="horizontal")

    for clause_node in list(clause_graph.nodes):
        ranges = clause_graph.nodes[clause_node]["range"]
        datas = [compound_extractor.extract_pyltp(words[s:e], postags[s:e]) for s, e in ranges]
        if all([len(data) == 0 for data in datas]): continue
        indices = ut.where(datas, lambda _, x: "compound" in x)
        if len(indices) != 1:
            raise ValueError("More than one compound extracted.")
        clause_graph = split_node(clause_graph, clause_node, compound_splitter_wrapper(datas),
                                  k=len(datas[indices[0]]["compound"]), mode="vertical")

    return clause_graph


def transform_compound_node(clause_graph, words, postags, arcs, roles):
    """
    This function is mainly used to find compound clauses. Compound clauses mean that two clauses play a similar roles
    with respect to causality. In this function, clause_graph will be transformed as a tree-structured object which
    is also the ultimate cause and effect graph.

    Note we have two rules applied here. Compound rule decide whether two clauses are compound. And succ rule decide
    which clause is the end node of this compound relation.
    :param clause_graph: networkx.DiGraph object to represent a clause tree.
    :param words: word list
    :param postags: pos_tag list
    :param arcs: [arc.head, arc.relation] list
    :param roles: [role.index, role.arguments]; arguments: [arg.name, arg.range.start, arg.range.end]
    :return:
        clause_graph: networkx.DiGraph object to represent a clause tree.
    """
    # adjust the graph according to conj

    # use a queue to traverse all nodes in a directed order

    node_queue = queue.Queue()
    for node in filter(lambda n: clause_graph.in_degree[n] == 0, clause_graph):
        node_queue.put(node)

    # BFS to traverse all nodes in clause_graph
    visited = set()
    while not node_queue.empty():
        node = node_queue.get()
        visited.add(node)
        if __compound_rule(clause_graph, node, words, postags, arcs, roles):
            # if compound, we find the succ node and traverse the following nodes
            clause_graph = parallelize_node(clause_graph, node, node)
        for s_node in clause_graph.succ[node]:
            if s_node in visited: continue
            node_queue.put(s_node)

    return clause_graph


def build_sentence_graph(words, postags, arcs, roles):
    """
    This function is used to build concrete graph for paratactic sentence. We introduce three concepts about this
    function: `segment`, `short`, `clause`. A `segment` is one word sequence that ends with comma. `Short` means
    short sentence embodied in the long sentence. `Clause` means some semantic unit extracted from `short`.
    We build segment tree using parsed tree and merge segments based on __merge_rule rule. Merged segments should be
    short sentences. Some extra-sentence conjunction words are used to transform short graph structure. Clauses are
    extracted from short sentence via `causality_extractor` and `compound_extractor`. In the end, we remove some
    redundant nodes.

    :param words: word list
    :param postags: pos_tag list
    :param arcs: [arc.head, arc.relation] list
    :param roles: [role.index, role.arguments]; arguments: [arg.name, arg.range.start, arg.range.end]
    :return:
        clause_graph: networkx.DiGraph object to represent a clause tree.
    """
    segment_graph = build_segment_tree(words, postags, arcs, roles)

    clause_graph = merge_segments(segment_graph, words, postags, arcs, roles)

    clause_graph = transform_compound_node(clause_graph, words, postags, arcs, roles)

    clause_graph = extract_clauses(clause_graph, words, postags, arcs, roles)

    return clause_graph


"""
Problem:
1. 并列时间问题: 顺承?因果?并列
2. 节点间并列判断与因果/节点内并列抽取的先后顺序问题:
    A, cc B  : 这种默认B和前面所有的最终节点并列
    A cc B,  : 这种默认B之和A并列
3. 删除无用节点: 目前只删除了 SBV + Psych-verb 形式的内容
4. 由于默认存在前句到后句的顺承关系,所以当流水句表意完全不相关的事情时会有多余的顺承关系
"""

if __name__ == '__main__':
    # run()

    sents = [
        # "对岸官方去年大力度实施“错峰生产”及提高环保要求，产能受控下，市场量稳价增，各厂首季淡季转旺，第2、3季出货价也维持高档，第4季配合最旺季，市场价格来到全年最高峰，据中国水泥网统计数据，全大陆P42.5"
        # "散装水泥每公吨均价已超过人民币400元，超过近16年均价近人民币百元，今年在官方去产能力度更大情况下，国内西进业者对明年水泥景气具高度信心。",
        # "今年台湾水泥市场需求可能持续低迷，但水泥价格有机会较去年持稳甚至小增，中国市场仍是台泥的主要成长力道，而和平电厂方面也可望受惠于电价调升，带动获利向上，台泥全年获利成长力道将胜往年。",
        # "台泥国际已从港股完成下市私有化，从12月起台泥在大陆市场的营收获利可100%全数贡献给台泥，预期去年第四季营收、获利有机会同步冲高",
        # "台泥在长三角供不应求，两广地区市场出货价逐步增温下，2017年12月销售可望达2017年高峰，并延续到2018年1月、甚至2月上旬。",
        # "今年台湾水泥市场需求可能持续低迷，但水泥价格有机会较去年持稳甚至小增，中国市场仍是台泥的主要成长力道，而和平电厂方面也可望受惠于电价调升，带动获利向上，台泥全年获利成长力道将胜往年",
        # "从产业面来看，苯酚市场供需持续改善中，过去由于产能过剩加上下游CPL及PC产业状况不佳，一直呈现供过于求状态,"
        # "未来产业上由于环己酮改以苯酚进料趋势确立，将会更加巩固苯酚的用量;法人表示，大陆近两年持续增加CPL自给率，新增近百万吨产能，且近期CPL需求状况不错，可望带动上游原料环己酮需求;台泥此次处分持股后，持股比率仍有40"
        # "%，不致看淡信昌化未来营运发展",
        # "台泥(1101)受惠错峰生产及市场库存水位偏低，农历春节前价格修正幅度较小，1月营收年增逾5成，尽管2月有春节工作天数减半影响，预估营收仍维持年成长，3月在小旺季推动下，更有机会拉出一波涨价。由于台泥国际已于去年11"
        # "月底完成合并，12月起认列中国转投资收益大增，今年获利进一步冲高可期",
        # "展望2018年，目前大陆基础建设稳健，房地产需求平稳，但供给面有错峰生产、环保稽查两大因素，今年中国水泥市场营运有机会持续成长，加上和平电厂今年有机会受惠涨价，整体营运表现将胜去年",
        # "今日台泥受到利多新闻带动下，股价突破前高，创下波段新高，目前技术面上进入多方趋势，随着基本面的上扬，股价可望逐步垫高",
        # "近期中国水泥市场受到中国政府对于产能控制力道加大，水泥价格持续走扬，以台泥("
        # "1101)主要销售区域华东、两广及西南区域，平均售价大涨三成以上，加上台泥国际私有化，台泥将可以认列100%中国业绩，营运可望大幅成长，展望今年在中国整体产能受到控制下，水泥价格预估将居高不下",
        # "大陆官方今(2018)年将祭出持续去产能、强化错峰生产和环保及淘汰低阶水泥等三大措施，促使水泥市场秩序持续改善，有望延续去年景气向上荣景，西进水泥厂台泥(1101)、亚泥、信大及国产将高度受惠相关政策红利",
    ]
    for s in sents:
        a = 1
