import os
import re
import numpy as np
import utils as ut
from config import ABSTRACT_NODE_MAPPING_DIR, ANNOCOM_FEATURE_DIR
from database_model.loader import load_efficient_metas, load_stock_price_history, dump_instance_key
from predict.conf import AVERAGE_POOLING

"""     Build Instance Key      """


def date_shifter_wrapper():
    dates = sorted([re.search("\d+", filename).group()
                    for filename in ut.filter_list(lambda f: f.endswith("json"),
                                                   os.listdir(ABSTRACT_NODE_MAPPING_DIR))])
    date2id, id2date = ut.build_mapping_from_list(dates)

    def date_shifter(date, i):
        if i == 0: return date
        idx = date2id[date]
        return id2date.get(idx - i)

    return date_shifter


date_shifter = date_shifter_wrapper()


def annocom_parser_wrapper():
    metas = load_efficient_metas()

    announcement2datetime = {meta.announcement_id: meta.datetime for meta in metas}

    def annocom_parser(annocom):
        s = annocom.split(":")
        company = s[1]
        announcement_id = s[0]
        original_date = s[0].split("-")[0]
        hour = announcement2datetime.get(announcement_id).hour
        date = date_shifter(original_date, -1 if hour is not None and hour >= 25 else 0)
        return company, date if date is not None else original_date

    return annocom_parser


annocom_parser = annocom_parser_wrapper()


def build_instance_key(K=3):
    with open(os.path.join(ANNOCOM_FEATURE_DIR, "featured_annocoms.txt"), "r", encoding="utf-8") as f:
        featured_annocoms = [l.strip() for l in f.readlines() if len(l) > 5]
    print("Start to build instance key.")

    instance2annocoms_on_date = {k: set(v) for k, v in ut.group_by_key(
        featured_annocoms,
        key=annocom_parser,
        value=lambda e: e
    ).items()}

    instance2annocoms = {
        instance: set.union(*[
            instance2annocoms_on_date[(instance[0], dt)]
            for dt in [date_shifter(instance[1], i) for i in range(K)]
            if (instance[0], dt) in instance2annocoms_on_date
        ]) for instance in instance2annocoms_on_date
    }

    dump_instance_key(instance2annocoms)
    print("Instance key is dumped into database.")

    return True


"""     Extract Instance Feature      """


def extract_instance_price_feature(row, hist_price_range=(3, 8)):
    s, e = hist_price_range
    prices = row[s:e]
    change = [(next - prev)/prev for prev, next in zip(prices[:-1], prices[1:]) if
              prev is not None and next is not None]
    return np.array([min(change), max(change), sum(change)/len(change),
                     len([chg for chg in change if chg > 0]),
                     len([chg for chg in change if chg < 0])])


def extract_instance_annocom_feature(row, annocom_feature, annocom_mapping, annocoms_idx=-1):
    annocoms = [annocom_mapping[0][anno] for anno in row[annocoms_idx].split(",")]
    identifier = np.zeros((annocom_feature.shape[0],))
    identifier[annocoms] = (1.0/len(annocoms)) if AVERAGE_POOLING else 1.0
    return np.matmul(identifier, annocom_feature)


def extract_instance_price_target(row, future_price_range=(8, 13), yesterday=7, today=8):
    s, e = future_price_range
    price = row[s:e]
    price_prev = row[yesterday]
    price_curr = row[today]
    change = [
                 (p - price_prev)/price_prev if p is not None else None for p in price
             ] + [
                 (p - price_curr)/price_curr if p is not None else None for p in price[1:]
             ]
    return np.array(change)


def extract_instance_feature():
    """
    This function integrate announcement features to instance feature.
    :return:
    """
    annocom_feature = np.load(os.path.join(ANNOCOM_FEATURE_DIR, "annocom_feature.npy"))

    with open(os.path.join(ANNOCOM_FEATURE_DIR, "featured_annocoms.txt"), "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if len(l) > 5]
        annocom_mapping = ut.build_mapping_from_list(lines)

    data = load_stock_price_history()
    print("Load instance key from database.")

    instance_feature = np.zeros((len(data), 5 + annocom_feature.shape[1]))
    instance_target = np.zeros((len(data), 5 + 4))
    ids = []
    for i, row in enumerate(data):
        ids.append(row[0])
        price_feature = extract_instance_price_feature(row)
        anno_feature = extract_instance_annocom_feature(row, annocom_feature, annocom_mapping)
        instance_feature[i] = np.hstack((price_feature, anno_feature))
        instance_target[i] = extract_instance_price_target(row)
    print("Finish extracting instance features.")

    np.save(os.path.join(ANNOCOM_FEATURE_DIR, "instance_feature.npy"), instance_feature)
    np.save(os.path.join(ANNOCOM_FEATURE_DIR, "instance_target.npy"), instance_target)
    with open(os.path.join(ANNOCOM_FEATURE_DIR, "featured_instance.txt"), "w", encoding="utf-8") as f:
        f.writelines(["{}\n".format(id) for id in ids])
    print("All instance features are written in {}.".format("instance_feature.npy"))
    print("All instance targets are written in {}.".format("instance_target.npy"))
    print("Instance index is written in {}.".format("featured_instance.npy"))

    return True


"""     Load Instance Feature      """


def load_instance():
    instance_feature = np.load(os.path.join(ANNOCOM_FEATURE_DIR, "instance_feature.npy"))
    instance_target = np.load(os.path.join(ANNOCOM_FEATURE_DIR, "instance_target.npy"))
    return instance_feature, instance_target


if __name__ == '__main__':
    # build_instance_key()
    extract_instance_feature()
