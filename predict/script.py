import json
import os

import utils as ut
from config import *
from database_model.ext import DBSession
from database_model.loader import load_efficient_metas, load_stock_price_history
from predict.annocom_feature import extract_annocom_feature, extract_event_embedding
from predict.conf import DISCRETE_BETA
from predict.instance_feature import build_instance_key, extract_instance_feature
from text.abstract_graph.feature_utils import load_event_feat, load_vocab_mappings


def c(a, b):
    if a is None or b is None:
        return None
    return (b - a)/a


def annocom_to_price():
    # announcement
    metas = load_efficient_metas()
    announcement2headline = {meta.announcement_id: meta.headline for meta in metas}
    announcement2datetime = {meta.announcement_id: meta.datetime for meta in metas}

    # stock price for instance
    data = load_stock_price_history()
    announcement2rows = ut.group_by_key([
        (i, annocom.split(":")[0]) for i, row in enumerate(data) for annocom in row[-1].split(",")
    ], key=lambda tpl: tpl[1], value=lambda tpl: tpl[0])

    # event description
    id2event, event_feats = load_event_feat(load_vocab_mappings())

    announcement2events = {}
    # announcement event mapping
    filenames = ut.filter_list(lambda f: f.endswith("json"), os.listdir(HEADLINE_NODE_MAPPING_DIR))
    for filename in filenames:
        with open(os.path.join(HEADLINE_NODE_MAPPING_DIR, filename), "r", encoding="utf-8") as f:
            overall_node_mappings = json.load(f)
        announcement2events.update(overall_node_mappings)

    session = DBSession()
    drop_sql = "DROP TABLE IF EXISTS `annocom_price_hist`;"
    create_sql = """
      CREATE TABLE `annocom_price_hist` (
      `id` int(16) DEFAULT NULL,
      `company_code` varchar(64) DEFAULT NULL,
      `date` varchar(64) DEFAULT NULL,
      `datetime` datetime DEFAULT NULL,
      `annocoms` text DEFAULT NULL,
      `announcement_id` varchar(64) DEFAULT NULL,
      `annocom_id` varchar(64) DEFAULT NULL,
      `headline` text DEFAULT NULL,
      `events` text DEFAULT NULL,
      `price_5` float DEFAULT NULL,
      `price_4` float DEFAULT NULL,
      `price_3` float DEFAULT NULL,
      `price_2` float DEFAULT NULL,
      `price_1` float DEFAULT NULL,
      `price_p0` float DEFAULT NULL,
      `price_p1` float DEFAULT NULL,
      `price_p2` float DEFAULT NULL,
      `price_p3` float DEFAULT NULL,
      `price_p5` float DEFAULT NULL,
      `change_0` float DEFAULT NULL,
      `change_1` float DEFAULT NULL,
      `change_2` float DEFAULT NULL,
      `change_3` float DEFAULT NULL,
      `change_5` float DEFAULT NULL,
      `change_p1` float DEFAULT NULL,
      `change_p2` float DEFAULT NULL,
      `change_p3` float DEFAULT NULL,
      `change_p5` float DEFAULT NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    session.execute(drop_sql)
    session.flush()
    session.execute(create_sql)
    session.commit()

    concent_list = [
        """
        (
            {id}, '{company_code}', '{date}', '{datetime}', '{annocoms}', '{announcement_id}',
            '{annocom_id}', '{headline}', '{events}',
            {price_5}, {price_4}, {price_3}, {price_2}, {price_1}, {price_p0},
            {price_p1}, {price_p2}, {price_p3}, {price_p5},
            {change_0}, {change_1}, {change_2}, {change_3}, {change_5},
            {change_p1}, {change_p2}, {change_p3}, {change_p5}
        )
        """.format(
            id=data[row_id][0], company_code=data[row_id][2], date=data[row_id][1],
            datetime=announcement2datetime.get(announcement_id),
            annocoms=data[row_id][13],
            announcement_id=announcement_id,
            annocom_id=":".join([announcement_id, data[row_id][2]]),
            headline=announcement2headline.get(announcement_id),
            events=",".join([
                "{}:{}".format(event, id2event[event])
                for event in announcement2events[announcement_id][data[row_id][2]]
            ]) if announcement_id in announcement2events
                  and data[row_id][2] in announcement2events[announcement_id]
            else "",
            price_5=data[row_id][3], price_4=data[row_id][4], price_3=data[row_id][5],
            price_2=data[row_id][6], price_1=data[row_id][7], price_p0=data[row_id][8],
            price_p1=data[row_id][9], price_p2=data[row_id][10], price_p3=data[row_id][11],
            price_p5=data[row_id][12],
            change_0=c(data[row_id][7], data[row_id][8]),
            change_1=c(data[row_id][7], data[row_id][9]),
            change_2=c(data[row_id][7], data[row_id][10]),
            change_3=c(data[row_id][7], data[row_id][11]),
            change_5=c(data[row_id][7], data[row_id][12]),
            change_p1=c(data[row_id][8], data[row_id][9]),
            change_p2=c(data[row_id][8], data[row_id][10]),
            change_p3=c(data[row_id][8], data[row_id][11]),
            change_p5=c(data[row_id][8], data[row_id][12]),
        ).replace("None", "null") for announcement_id, row_ids in announcement2rows.items() for row_id in row_ids
    ]
    for i in range(0, len(concent_list), 1000):
        print("Finish ", i)
        insert_sql = "INSERT INTO `annocom_price_hist` VALUES {};".format(
            ",".join(concent_list[i:i + 1000])
        )
        session.execute(insert_sql)
        session.flush()
    session.commit()
    session.close()
    return


def whole_procee():
    extract_event_embedding(beta=0)
    extract_annocom_feature(node_mapping_mode="headline")
    build_instance_key(K=1)
    extract_instance_feature()
    return


if __name__ == '__main__':
    whole_procee()
    annocom_to_price()
