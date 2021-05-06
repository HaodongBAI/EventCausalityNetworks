import json
from sqlalchemy import and_, func, not_

import utils as ut
from database_model.ext import DBSession
from database_model.text_model import Announcement
from config import ANNOUNCEMENT_MAIN_DIR


def load_metas(stock_code):
    session = DBSession()
    anno_metas = session.query(Announcement).filter(
        and_(
            Announcement.stock_code.contains(stock_code),
        )).all()
    return anno_metas


def load_story_from_metas(metas):
    # 按照Announcement记录读入json文件
    anno_id_file = [(meta.announcement_id, "announcement-" + meta.file_path.split("/")[0] + ".json") for meta in metas]
    anno_file2ids = {k: set(v) for k, v in
                     ut.group_by_key(anno_id_file, key=lambda t: t[1], value=lambda t: t[0]).items()}
    anno_id2story = {}
    for file, ids in anno_file2ids.items():
        js_objs = load_story_from_filename(filename=file)
        anno_id2story.update(
            {js_obj["announcement_id"]: js_obj["story"] for js_obj in js_objs if js_obj["announcement_id"] in ids})
    return anno_id2story  # dict of {announcement_id: story}


def load_story_from_filename(filename):
    with open(ANNOUNCEMENT_MAIN_DIR + filename, "r",
              encoding="utf-8") as f:
        js_objs = json.loads(f.read())

    """
    js_objs = [{
        "announcement_id": str, 
        "story": str 
    },]
    """
    return js_objs


def load_story_from_announcement_id(announcement_id):
    session = DBSession()
    anno_metas = session.query(Announcement).filter(
        and_(
            Announcement.announcement_id == announcement_id,
        )).all()
    stories = load_story_from_metas(anno_metas)
    if announcement_id not in stories:
        ValueError("No story retrieved using announcement_id: {}.".format(announcement_id))
    return stories[announcement_id]


def load_meta_story_price(stock_code):
    # 读入Announcement记录/新闻内容/股价走势
    anno_metas = load_metas(stock_code)

    session = DBSession()
    anno_id2story = load_story_from_metas(anno_metas)

    date_price = session.execute(f"""
        select date, close_price 
        from tmp_stock_price_d_rn
        where company_code = "{stock_code}"
    """).fetchall()
    session.close()
    return anno_metas, anno_id2story, date_price


def load_efficient_metas(filedate=None):
    cond = and_(
        Announcement.tag.in_({
            "法人動態", "行庫動態", "電子時報", "財訊快報", "焦點新聞", "焦點類股",
            "個股", "個股解析", "焦點股", "熱門股", "蘋果日報頭條", "工商證券頭條", "工商時報頭條", "經濟證券頭條", "經濟日報頭條"
        }),
        not_(Announcement.author == "證交所")
    )

    if filedate is not None:
        cond = and_(cond, Announcement.announcement_id.contains(filedate))
    session = DBSession()
    anno_metas = session.query(Announcement).filter(cond).all()
    session.close()
    return anno_metas


def dump_instance_key(instance2announcements):
    session = DBSession()

    session.execute("DROP TABLE IF EXISTS `instance_key`;")
    session.flush()

    session.execute("""
    CREATE TABLE `instance_key` (
      `id` int(16) DEFAULT NULL,
      `company_code` varchar(64) DEFAULT NULL,
      `date` varchar(64) DEFAULT NULL,
      `announcements` text DEFAULT NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    session.flush()

    session.execute("INSERT INTO `instance_key` VALUES {};".format(
        ",".join(["({}, '{}', '{}', '{}')".format(
            str(i), instance[0], instance[1], ",".join(announcements)
        ) for i, (instance, announcements) in enumerate(sorted(instance2announcements.items(), key=lambda t: t[0]))])
    ))
    session.commit()

    session.execute("DROP TABLE IF EXISTS `instance_stock_price_hist`;")
    session.flush()

    session.execute("""
    CREATE TABLE instance_stock_price_hist AS 
    SELECT b.id, c.rn, a.date, a.company_code, 
            a.close_price_5 price_5, 
            a.close_price_4 price_4,
            a.close_price_3 price_3,  
            a.close_price_2 price_2,
            a.close_price_1 price_1, 
            a.close_price price_p0, 
            d.close_price price_p1, 
            e.close_price price_p2,
            f.close_price price_p3, 
            g.close_price price_p5,
            b.announcements
    FROM tmp_stock_price_d_hist a JOIN instance_key b 
        ON a.company_code = b.company_code AND a.date = b.date 
    JOIN tmp_stock_price_d_rn c 
        ON c.company_code = b.company_code AND c.date = b.date 
    JOIN tmp_stock_price_d_rn d
        ON c.company_code = d.company_code AND c.rn = d.rn_1
    JOIN tmp_stock_price_d_rn e
        ON c.company_code = e.company_code AND c.rn = e.rn_2
    JOIN tmp_stock_price_d_rn f
        ON c.company_code = f.company_code AND c.rn = f.rn_3
    JOIN tmp_stock_price_d_rn g
        ON c.company_code = g.company_code AND c.rn = g.rn_5
    WHERE a.close_price_1 IS NOT NULL AND a.close_price IS NOT NULL
    ORDER BY b.id
    """)
    session.commit()

    session.close()
    return True


def load_stock_price_history():
    session = DBSession()
    data = session.execute("""
        SELECT id, date, company_code,
            price_5,price_4,price_3,price_2,price_1,
            price_p0,price_p1,price_p2,price_p3,price_p5,
            announcements
        FROM instance_stock_price_hist
    """).fetchall()
    session.close()
    return data
