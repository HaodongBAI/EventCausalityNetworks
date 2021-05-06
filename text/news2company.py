# coding=utf-8
import json
from  functools import reduce
from itertools import chain

from database_model.ext import DBSession
from database_model.text_model import Announcement

class NewsClassification:
    """
    top 表示最高层的标签
    tag 表示对head的基本处理精简,底层标签
    head 原生的从标题中提取的内容
    """
    vocab = [
        "個股",
        "個股解析",
        "自營商個股",
        "外資個股",
        "投信個股",

        "產業",
        "營收",
        "盈餘",
        "基金",
        "財報",
        "櫃買中心",
        "融資券",
        "焦點股",
        "熱門股",

        "股東會",
        "董事會",
        "法人觀點",
        "集中市場",
        "自營商",
        "投信",
        "外資",
        "經濟指標",
        "除權息",
        "除權",
        "海外存託憑證",

        "警示股",
        "科技股",
        "外資觀點",
        "先探投資週刊",
        "法說會",

        "國內債市",
        "國內匯市",
        "內匯市",
        "大陸股市",
        "大陸匯市",
        "大陸金融",
        "國內金融",
        "國際股市",
        "國際債市",
        "國際匯市",
        "國際金融",
        "台股期貨",
        "期貨",

        "美股",
        "歐股",
        "亞洲股市",
        "陸股",
        "日股",
        "韓股",
        "澳股",
        "亞股",
        "星股",
        "美股電子盤",
        "港股",

        "上市申報轉讓",
        "上櫃申報轉讓",
        "興櫃申報轉讓",

        "準上市",
        "未上市",
        "準上櫃",
        "上櫃",
        "興櫃",

        "金價",
        "地產",
        "油價",
        "物料",
        "電子市場",
        "營建",
        "房地產",
        "大宗物資",
        "宗物資",
        "原物料",

        "元富台指期貨評論",
        "永豐台指期貨評論",
        "統一台指期貨評論",
        "國票台指期貨評論",
        "統一台指數期貨評論",
        "永豐台股期貨評論",

        "摩台電",
        "摩台期",

        "晨間解析",
        "統一晨盤",
        "大昌晨盤",
        "德信晨盤",
        "大慶晨盤",
        "福邦晨盤",
        "亞東晨盤",
        "國票晨盤",
        "大展晨盤",
        "日盛晨盤",

        "盤中分析",
        "盤後分析",
        "美股盤後",
        "盤後交易",

        "外資動態",
        "外銷訂單",

        "休市通知",
        "產業隊長•張捷",
        "法人動態",
        "行庫動態",
        "重大訊息",
        "權證",
        "更正",
        "紐約金",
        "外存託憑證",
        "董事長",
        "證期局",

        "精誠資訊",
        "技業資訊",
        "精爵資訊",

        "國際產經",
        "國際財經",
        "大陸產經",
        "大陸財經",
        "國內產經",
        "國內財經",

        "國際產經評論",
        "大陸產經評論",
        "國際財經評論",
        "蘋果財經頭條",
        "大陸財經評論",

        "國際股市評論",
        "大陸股市評論",

        "蘋果日報頭條",
        "工商證券頭條",
        "工商時報頭條",
        "經濟證券頭條",
        "經濟日報頭條",
        "電子時報",
        "財訊快報",

        "大陸要聞",
        "國內要聞",
        "國際要聞",
        "股市要聞",
        "焦點新聞",
        "焦點類股",

    ]
    top2head = {
        "个股财讯"   : ["個股", "營收", "盈餘", "財報", "個股解析", "重大訊息"],
        "未上市个股财讯": [
            "準上市", "未上市", "準上櫃", "上櫃", "興櫃", "上市申報轉讓", "上櫃申報轉讓", "興櫃申報轉讓"
        ],
        "董事股东会"  : ["股東會", "董事會"],
        "除权除息"   : ["除權息", "除權"],

        "警示焦点股"  : ["焦點股", "警示股", "熱門股"],  # 还需要增加判断,剔除 "买超卖超" 買超／賣超

        "物资产业"   : ["產業", "金價", "地產", "油價", "物料", "電子市場", "營建", "房地產", "大宗物資", "宗物資", "原物料"],

        "台股大盘"   : [
            "晨間解析", "統一晨盤", "大昌晨盤", "德信晨盤", "大慶晨盤", "福邦晨盤", "亞東晨盤", "國票晨盤", "大展晨盤", "日盛晨盤",
            "盤中分析", "盤後分析", "盤後交易", "集中市場", "櫃買中心", "融資券",
        ],

        "台股其他参与人": [
            "自營商個股", "自營商", "投信個股", "投信", "外資", "外資個股", "基金", "法說會", "證期局", "海外存託憑證"
        ],

        "债汇及外部市场": [
            "國內債市", "國內匯市", "大陸股市", "大陸匯市", "大陸金融",
            "國內金融", "國際股市", "國際債市", "國際匯市", "國際金融", "台股期貨", "期貨",
            "科技股", "美股", "美股盤後", "歐股", "亞洲股市", "陸股", "日股", "韓股",
            "澳股", "亞股", "星股", "美股電子盤", "港股", "摩台電", "摩台期"  # 摩台 系 新加坡市场
        ],

        "观点评论"   : [
            "法人觀點", "外資觀點",
            "國際產經評論", "大陸產經評論", "國際財經評論", "蘋果財經頭條", "大陸財經評論", "國際股市評論", "大陸股市評論",
            "元富台指期貨評論", "永豐台指期貨評論", "統一台指期貨評論", "國票台指期貨評論", "統一台指數期貨評論", "永豐台股期貨評論",
        ],

        "要闻头条"   : [
            "法人動態", "行庫動態",
            "經濟指標",  # 也是一类似新闻,主要讲美国情况
            "先探投資週刊",
            "精誠資訊", "技業資訊", "精爵資訊",
            "國際產經", "國際財經", "大陸產經", "大陸財經", "國內產經", "國內財經",
            "蘋果日報頭條", "工商證券頭條", "工商時報頭條", "經濟證券頭條", "經濟日報頭條",
            "電子時報", "財訊快報",
            "大陸要聞", "國內要聞", "國際要聞", "股市要聞",
            "焦點新聞", "焦點類股",
        ],
    }
    head2top = {v: k for k, vv in top2head.items() for v in vv}
    list_head = [
        "基金", "櫃買中心", "集中市場", "融資券", "期貨", "股東會", "美股",
        "熱門股", "除權息", "盤後交易", "外資個股", "投信個股", "自營商個股", "自營商",
        "法說會", "上市申報轉讓", "上櫃申報轉讓", "興櫃申報轉讓",
    ]


# 对标题进行预处理
def head_extractor(line):
    colon_index = line.find("：")
    if colon_index == -1:
        # 替换掉 要闻头条中的字串
        head_candidate = reduce(
            lambda cand, tag: tag if tag in line else cand,
            NewsClassification.top2head["要闻头条"], None
        )
        if head_candidate is None: return head_candidate
    else:
        head_candidate = line[:colon_index]

    head_candidate = reduce(lambda h, i: i if i in h else h, [
        "先探投資週刊", "財訊快報", "精誠資訊"
    ], head_candidate)

    # 删除这些字符
    head_candidate = reduce(lambda h, r: h.replace(r, ""), [
        "▲", "▼", "△", "▽"
    ], head_candidate)

    return head_candidate


# 从标题中找到tag和top
def headline_tagger(line):
    head = head_extractor(line)
    list_predicate = lambda l: "一覽表" in l or ("前" in l and "名" in l)

    if head and head in NewsClassification.list_head and list_predicate(line):
        tag = head+"-一覽表"
    else:
        tag = head
    return tag, NewsClassification.head2top.get(head)


# 入口函数 确定文本的特征
def news_classification():
    session = DBSession()
    announcements = session.query(Announcement).all()

    for a in announcements:
        tag, top = headline_tagger(a.headline)
        a.tag = tag
        a.top = top

    session.commit()
    session.close()
    return


def anno_com_align_logic(company, headline, story):
    res = []
    headline = headline.replace(" ", "")
    story = story.replace(" ", "")
    for (cd, short, full) in company:
        if short in headline or short in story or full in headline or full in story:
            res.append(cd)
            continue
    return res


def merge_codes(stock_codes, main_code, text_codes):
    code_in_db = []
    if stock_codes is not None: code_in_db.extend(stock_codes.split(","))
    if main_code is not None: code_in_db.append(main_code)
    return ",".join(set(chain(
        map(lambda cd: cd.replace("N", ""), code_in_db),
        text_codes)))


def anno_com_align():
    session = DBSession()
    com_cd_short_full = session.execute("""
        select `代碼`,`股票名稱`,`公司名稱`
        from `公司基本信息表`
        where `代碼` is not null
    """).fetchall()

    for (date,) in session.execute("""
        select distinct substr(announcement_id,1,8)
        from announcement
    """).fetchall():
        anno_metas = session.query(Announcement).filter(
            Announcement.announcement_id.contains(date)
        ).all()
        with open("announcement-json/announcement-"+date+".json", "r", encoding="utf-8") as f:
            anno_id2story = {anno_story["announcement_id"]: anno_story["story"] for anno_story in json.loads(f.read())}
        for anno_meta in anno_metas:
            anno_id = anno_meta.announcement_id
            anno_headline = anno_meta.headline
            anno_story = anno_id2story[anno_id]
            text_codes = anno_com_align_logic(com_cd_short_full, anno_headline, anno_story)
            anno_meta.stock_code_self = merge_codes(anno_meta.stock_code, anno_meta.main_stock, text_codes)
        print(f"Finish date {date}...")
        session.commit()
    session.close()
    return


if __name__ == '__main__':
    anno_com_align()
