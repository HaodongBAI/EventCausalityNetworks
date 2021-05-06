from itertools import accumulate, combinations, product

import networkx as nx
import networkx.algorithms.bipartite as bi
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

import utils as ut
from concrete_graph.conf import PERCENTILE_SIMILARITY_VERBOSE
from config import LTP_MODEL_DIR

with open(LTP_MODEL_DIR+"/stoplist", "r", encoding="utf-8") as f:
    stop_list = {w.strip() for w in f.readlines()}


def clause_similarity(nested_clauses, nested_clause_names, stop_list=None, percentile=0.9):
    """
    This function accepts clauses over all sentences in one announcement and gives a list of clause clusters. Clauses
    in one cluster will be merged in following graph fusion process. Firstly,
    we use sklearn.feature_extraction.text.CountVectorizer to vectorize clauses in these sentences. Similarity
    between clauses are obtained by normalized inner product of clause vectors. Then in order to avoid to merge
    clauses in one sentences, we construct Bipartite Maximum Matching problem. At last, we use UnionFind to organize
    all clauses clusters.


    :param nested_clauses: list[list[list[str]]] list of sentences, list of clauses, list of words.
    :param nested_clause_names: list[list[str/int]] node name corresponding to each clause in `nested_clauses`
    :param stop_list: stop word list
    :param percentile: clauses with similarity larger than `percentile` will be denote as similar.
    :return:
        clause_clusters: list of node lists. Each node list contains nodes in one cluster.
    """

    # Vectorize clauses using CountVectorizer
    clause_names = [(i, n) for i, l in enumerate(nested_clause_names) for n in l]
    clauses = [" ".join(c) for cs in nested_clauses for c in cs]
    vectorizer = CountVectorizer(stop_words=stop_list, ngram_range=(1, 1))

    clause_vectors = vectorizer.fit_transform(clauses).toarray()

    # Similarity is normalized inner product of count vectors.
    occurance = np.matmul(clause_vectors, clause_vectors.T)
    similarity = np.zeros(occurance.shape)
    for i, j in product(range(len(clauses)), repeat=2):
        if occurance[i, j] == 0:
            similarity[i, j] = 0
        else:
            similarity[i, j] = (1.0 * occurance[i, j]) / (occurance[i, i]+occurance[j, j]-occurance[i, j])

    efficient_sim = similarity[(similarity < 1) & (similarity > 0)]
    if efficient_sim.shape[0] == 0:
        # if there is no similar clauses
        return [[node_name_tpl, ] for node_name_tpl in clause_names]

    if PERCENTILE_SIMILARITY_VERBOSE:
        p = np.percentile(efficient_sim, q=[q for q in range(50, 100, 5)])
        print("Similarity Percentile: {}.".format(
            "; ".join(["%i%%:%.2f" % (q, per) for q, per in zip(range(50, 100, 5), p)])))

    p = max(np.percentile(efficient_sim, q=[percentile * 100, ])[0], 0.25)
    boolean_sim = (similarity > p).astype(int)

    # Use union-find method to organize nodes to merge
    union_find = ut.UnionFind(clause_names)

    # Map sentence index to clause index range, the i-th sentence has clauses ranging over sentence_ranges[i]
    s = list(accumulate([len(c) for c in nested_clauses]))
    sentence_ranges = list(zip([0]+s[:-1], s))

    for i, j in combinations(range(len(sentence_ranges)), 2):
        # Assume sentence i has n clauses, sentence j has m clauses
        (s_i, e_i), (s_j, e_j) = sentence_ranges[i], sentence_ranges[j]
        adjacency_mat = boolean_sim[s_i:e_i, s_j:e_j]
        g = bi.from_biadjacency_matrix(sp.coo_matrix(adjacency_mat))
        matching = nx.max_weight_matching(g)
        for l, r in map(lambda tpl: tpl if tpl[0] < tpl[1] else (tpl[1], tpl[0]), matching):
            union_find.union((i, nested_clause_names[i][l]),
                             (j, nested_clause_names[j][r-len(nested_clause_names[i])]))

    # The output is a list of node lists. Each node list contains nodes in one cluster.
    # Each clause node is denoted as a tuple where the first entry is sentence index and the second is node name.
    # [[(1, 2), (2, 4), (3, 2)], [(1, 3)], [(2, 2)], [(2, 1), (3, 1)]]
    return list(union_find.root2ele.values())


def node_merger_wrapper(nested_clause_names, nested_clauses):
    node2clause = {(sent_i, node): clause for sent_i, (nodes, clauses) in
                   enumerate(zip(nested_clause_names, nested_clauses))
                   for node, clause in zip(nodes, clauses)}

    def node_merger(nodes):
        node2data = {"clause": ["".join(node2clause[node_name]) for node_name in nodes]}
        return node2data

    return node_merger


def fuse_clause_graphs(clause_graphs, node_clusters, node_merger):
    """
    This function is used to fuse all graphs into one concrete graph. One new node is merged from a cluster in
    node_clusters by `node_merger`. The edges between cluster node will have an attribute `weight` to denote how many
    edges between nodes in clusters.

    :param node_clusters: output of clause_similarity; list of node lists. Each node list contains nodes in one cluster.
    :param clause_graphs: a list of clause graph, each clause graph corresponding to one sentence in given announcement
    :param node_merger: a function to deal with all node datas and give a dict as new node data
    :return:
    """
    g = nx.DiGraph()

    # build nodes
    node_data = ut.map_list(lambda nodes: node_merger(nodes), node_clusters)
    g.add_nodes_from(list(enumerate(node_data)))

    # build edges
    node_mapping = {n: i for i, ns in enumerate(node_clusters) for n in ns}
    for i, cg in enumerate(clause_graphs):
        for u, v in cg.edges():
            n1, n2 = node_mapping[(i, u)], node_mapping[(i, v)]
            if n1 == n2:
                continue
            elif g.has_edge(n1, n2):
                g[n1][n2]["weight"] += 1
            else:
                g.add_edge(n1, n2, weight=1)
    return g


if __name__ == '__main__':
    a = 1
