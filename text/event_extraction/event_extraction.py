import json
import os
import pyltp as ltp
import traceback
from collections import defaultdict

import utils as ut
from abstract_graph.feature_utils import FeatureUtils
from config import HEADLINE_FEATURE_DIR, LTP_MODEL_DIR
from database_model.loader import load_efficient_metas, load_story_from_metas
from preprocess import line_return_preprocess
import jieba


def extract_headline_feature():
    # segmentor = ltp.Segmentor()
    # segmentor.load_with_lexicon(LTP_MODEL_DIR + "/cws.model", LTP_MODEL_DIR + "/lexicon")

    jieba.load_userdict(LTP_MODEL_DIR + "/lexicon")

    feat_utils = FeatureUtils()

    metas = load_efficient_metas()

    id2metas = {meta.announcement_id: meta for meta in metas}
    filename2id = ut.group_by_key({id: id.split("-")[0] for id in id2metas}.items(),
                                  key=lambda tpl: tpl[1], value=lambda tpl: tpl[0])

    print("Load metas finished. There are {} files and {} announcements in total.".format(len(filename2id),
                                                                                          len(id2metas)))

    for filedate, ids in list(filename2id.items()):
        print("Processing filedate: {}, there are {} announcements in this file.".format(filedate, len(ids)))

        id2stoty = load_story_from_metas([id2metas[id] for id in ids])
        id2headline = {id: id2metas[id].headline for id in ids}
        id2clause_feature = defaultdict(dict)
        id2tag = {id: id2metas[id].tag for id in ids}

        for id in sorted(ids):
            if id == "20181012-00435":
                a = 0
            try:
                if id2tag.get(id) == "熱門股": continue
                story = id2stoty[id]
                headline = id2headline[id]
                headline = line_return_preprocess(headline)[0]

                if "：" in headline:
                    headline = "：".join(headline.split("：")[1:])
                sentences = [headline, ]

                # This is core logic
                # story_lines = line_return_preprocess(story)
                # sentences = headline + ut.filter_list(
                #     lambda s: len(s),
                #     map(
                #         lambda s: re.sub(r"【.*】", "", s).strip(),
                #         ut.flatten([line.split("。") for line in story_lines])
                #     )
                # )[:1]

                clauses = []

                for i, sent in enumerate(sentences):
                    comma = {"，", "；", "：", "，", "＋"}
                    segmented = jieba.lcut(sent)
                    # segmented = list(segmentor.segment(sent))
                    start, end = ut.split_list(segmented, lambda _, w: w in comma)
                    segments = [segmented[s:e] for s, e in zip(start, end)]
                    clauses.extend(
                        [
                            p + n for p, n in ut.window(segments, 2)
                            if len(p) < 10 or len(n) < 10
                        ] + segments)

                for clause_id, clause in enumerate(clauses):
                    word_set = {w1 + w2 for w1, w2 in zip(clause[:-1], clause[1:])}.union(clause)

                    feat_dict = {
                        "general_subject": feat_utils.general_subject_extractor(word_set),
                        "verb"           : feat_utils.verb_extractor(word_set),
                        "geographical"   : feat_utils.geographical_extractor(word_set),
                    }

                    id2clause_feature[id][clause_id] = {
                        "clause" : "|".join(clause),
                        "feature": feat_dict,
                        "length" : sum([len(v) for v in feat_dict.values()])
                    }

            except Exception as e:
                print("An error occurs when processing announcement {}".format(id))
                traceback.print_exc()
        with open(os.path.join(HEADLINE_FEATURE_DIR, "headline-{}.json".format(filedate)), "w", encoding="utf-8")as f:
            json.dump(id2clause_feature, f, indent=4, ensure_ascii=False)
    return True


if __name__ == '__main__':
    extract_headline_feature()
