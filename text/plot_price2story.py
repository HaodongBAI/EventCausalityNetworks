# coding=utf-8
import datetime

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import and_

import utils as ut
from database_model.ext import DBSession
from database_model.loader import load_meta_story_price, load_story_from_metas
from database_model.text_model import Announcement


def print_alignment(news_metas, id2story, prices):
    # 将新闻和股价走势对齐打印出来
    date2price = {datetime.datetime.strptime(p[0], "%Y%m%d").date(): float(p[1]) for p in prices if p[1] is not None}
    date2headlines = ut.group_by_key([(meta.datetime.date(), meta.headline) for meta in news_metas],
                                     key=lambda t: t[0], value=lambda t: t[1])
    date_price_headlines = [(dt, date2price[dt], date2headlines.get(dt, []))
                            for dt in sorted(date2price.keys()) if dt > datetime.date(2018, 1, 1)]

    for i, (dt, p, hs) in enumerate(date_price_headlines):
        gr = 100 * np.mean(ut.map_list(lambda t: t[1], date_price_headlines[i+1:i+11])) / p
        print(dt.strftime("%Y%m%d")+"  :  "+str(p), "%.2f" % gr)
        for h in hs:
            print("\t"+h)
        print("-------------------------------------------")
    plt.plot(ut.map_list(lambda t: t[0], date_price_headlines),
             ut.map_list(lambda t: t[1], date_price_headlines),
             "g*-")

    plt.show()
    return


def print_story(announcement_ids):
    session = DBSession()
    anno_metas = session.query(Announcement).filter(
        and_(
            Announcement.announcement_id.in_(announcement_ids)
        )).all()
    anno_id2story = load_story_from_metas(anno_metas)
    session.close()

    for anno_meta in sorted(anno_metas, key=lambda meta: meta.announcement_id):
        anno_id = anno_meta.announcement_id
        anno_date = anno_meta.datetime.date()
        anno_title = anno_meta.headline
        anno_story = anno_id2story[anno_id]
        anno_stock_code_self = anno_meta.stock_code_self
        print(anno_id, anno_date, anno_stock_code_self)
        print("标题:", anno_title)
        print()
        print(anno_story)
        print("============================================")
    return


if __name__ == '__main__':
    metas, id2story, prices = load_meta_story_price("1101.TW")
    print_alignment(metas, id2story, prices)
    # print_story(["20180102-00132", "20180102-00126"])
