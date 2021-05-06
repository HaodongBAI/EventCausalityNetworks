import pyltp as ltp

import utils as ut
from config import LTP_MODEL_DIR

"""     Subtree information rule      """


def first_level_nontrivial_adverbial_rule(root, sub_flag, words, postags, arcs, roles):
    return arcs[root].relation in {"SBV", "VOB", "IOB", "FOB", "DBL", "CMP", "POB"}


def nontrivial_adverbial_rule(root, sub_flag, words, postags, arcs, roles):
    return sub_flag or arcs[root].relation in {"SBV", "VOB", "IOB", "FOB", "DBL", "CMP", "POB"}


def nv_nominal_attribute_rule(root, sub_flag, words, postags, arcs, roles):
    return sub_flag or postags[root] in {"j", "n", "nd", "nh", "ni", "nl", "ns", "nz", "r", "ws", "v"}


def truncate_sub_tree(root, sub_flags, words, postags, arcs, roles):
    relation = arcs[root].relation
    if relation == "ADV" and not sub_flags["nontrivial_adverbial"]:
        return False
    elif relation == "ATT" and not sub_flags["nv_nominal_attribute"]:
        return False
    else:
        return True


"""     Compress       """


def compress_clause(words, postags, arcs, roles):
    iter2head = {i: arc.head-1 for i, arc in enumerate(arcs)}
    iter2children = ut.group_by_key(iter2head, key=lambda x: iter2head[x], value=lambda x: x)

    rules = {
        "nontrivial_adverbial": nontrivial_adverbial_rule,
        "nv_nominal_attribute": nv_nominal_attribute_rule
    }

    def traverse(root):
        """
        This function is used to traverse parsed tree of one clause. During the traverse, this function use the
        variable `flag` to deliver information of subtree. And `truncate_sub_tree` function accepts these information
        and decide whether to truncate the subtree.

        :param root: root node of sub_tree
        :return:
            indices: a list of nodes in a truncated tree.
            flag: a dict that the key is rule-name and the bool value denote whether these exists one subtree that
            meets the specific rule.
        """
        if root not in iter2children:
            return [], {rule: rule_func(root, False, words, postags, arcs, roles)
                        for rule, rule_func in rules.items()}

        sub_indices = list()
        sub_flags = list()

        for child in iter2children[root]:
            sub_index, sub_flag = traverse(child)

            if truncate_sub_tree(child, sub_flag, words, postags, arcs, roles):
                sub_indices.extend(sub_index+[child])

            # each sub_flag denote whether there exists one subtree of this child node meets the rule.
            # rule_func is used to judge whether child node and all its subtrees meets the rule.
            sub_flags.append({
                rule: rule_func(child, sub_flag[rule], words, postags, arcs, roles)
                for rule, rule_func in rules.items()
            })

        return sorted(sub_indices), {rule: any([sf[rule] for sf in sub_flags]) for rule in rules}

    indices, _ = traverse(-1)

    return [words[i] for i in sorted(indices)]


if __name__ == '__main__':
    segmentor = ltp.Segmentor()
    segmentor.load_with_lexicon(LTP_MODEL_DIR+"/cws.model", LTP_MODEL_DIR+"/lexicon")

    postagger = ltp.Postagger()
    postagger.load(LTP_MODEL_DIR+"/pos.model")

    parser = ltp.Parser()
    parser.load(LTP_MODEL_DIR+"/parser.model")

    srl = ltp.SementicRoleLabeller()
    srl.load(LTP_MODEL_DIR+"/pisrl.model")

    clauses = [
        "未来台湾水泥市场需求可能持续低迷",
        "场馆正在激烈紧张地举办一项高级赛事"
    ]
    for c in clauses:
        words = segmentor.segment(c)
        postags = postagger.postag(words)
        arcs = parser.parse(words, postags)
        roles = srl.label(words, postags, arcs)
        for i, (w, p, a) in enumerate(zip(words, postags, arcs)):
            print(i, w, p, a.relation, a.head-1, words[a.head-1])
        print("".join(compress_clause(words, postags, arcs, roles)))
