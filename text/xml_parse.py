# coding=utf-8
import json
import os
import re
import xml.etree.ElementTree as ET

from database_model.text_model import Announcement, News, Ptt
from database_model.ext import DBSession


def news_parse_unit(fp):
    tree = ET.parse(fp)
    article = tree.getroot()[0]
    content = {node.tag: node.text for node in article}
    return content


def announcement_parse_unit(fp):
    with open(fp, 'r', encoding="big5-hkscs") as f:
        s = f.read()
    s = s.replace("", "")  # 特殊字符不认识
    tree = ET.fromstring(s)
    # xmlp = ET.XMLParser(encoding="big5")
    # tree = ET.parse(fp, parser=xmlp)
    article = tree[0]
    content = {node.tag: node.text for node in article}
    content["story"] = article.find("story")[0].text
    return content


def ptt_parse_unit(fp):
    with open(fp, "r", encoding="utf-8") as f:
        s = "".join(list(f.readlines()))
    s = """<?xml version="1.0" encoding="UTF-8" ?><articles><article>""" \
        +s+"""</article></articles>"""
    s = s.replace("\n", "")
    s = re.sub(r"<title>(.*?)</title>",
               r"<title><![CDATA[\g<1>]]></title>",
               s, flags=re.M)
    s = re.sub(r"<content>(.*?)</content>",
               r"<content><![CDATA[\g<1>]]></content>",
               s, flags=re.M)
    # s = re.sub(r"<pushContent>(.*?)</pushContent>",
    #            r"<pushContent><![CDATA[\g<1>]]></pushContent>",
    #            s, flags=re.M)

    s = re.sub(r"<authot>(.*?)</author>",
               r"<author>\g<1></author>",
               s, flags=re.M)

    s = re.sub(r"<postTime>(.*?)<postTime>",
               r"<postTime>\g<1></postTime>",
               s, flags=re.M)

    tree = ET.fromstring(s)

    article = tree[0]
    content = article[6]
    push = article[7]

    ptt_dict = {node.tag: node.text for node in article[:6]}
    ptt_dict["content"] = content.text
    ptt_dict["push"] = {
        "pushTag"    : [node.text for node in push.findall("pushTag")],
        "pushTime"   : [node.text for node in push.findall("pushTime")],
        "pushUser"   : [node.text.strip() for node in push.findall("pushUser")],
        "pushContent": [node.text for node in push.findall("pushContent")]
    }
    ptt_dict["like_num"] = len(list(filter(lambda s: s == "推", ptt_dict["push"]["pushTag"])))
    ptt_dict["unlike_num"] = len(list(filter(lambda s: s == "噓", ptt_dict["push"]["pushTag"])))
    ptt_dict["null_num"] = len(list(filter(lambda s: s == "→", ptt_dict["push"]["pushTag"])))

    return ptt_dict


######################
def parse_loop_wrapper(parse_unit, obj_creator):
    def parse_loop(ifolder, file_selector=lambda fln, fn: True):
        js_objs = []
        db_objs = []
        for folder_name in sorted(os.listdir(ifolder)):
            if folder_name.startswith(".DS_Store"): continue
            for file_name in os.listdir(os.path.join(ifolder, folder_name)):
                if file_name.startswith(".DS_Store"): continue
                if not file_selector(folder_name, file_name): continue
                try:
                    c = parse_unit(os.path.join(ifolder, folder_name, file_name))

                    db_obj, js_obj = obj_creator(c, folder_name+"/"+file_name)
                    db_objs.append(db_obj)
                    js_objs.append(js_obj)
                except Exception as e:
                    print(e)
                    print(folder_name+"/"+file_name)
                    continue
            yield folder_name, db_objs, js_objs
            db_objs, js_objs = [], []

    return parse_loop


######################
def ptt_obj_creator(c, fp):
    c["ptt_id"] = fp.replace("/", "-")[:-4]
    c["file_path"] = fp
    return (Ptt.create(c), {
        "ptt_id" : c["ptt_id"],
        "content": c["content"],
        "push"   : c["push"]
    })


def news_obj_creator(c, fp):
    c["news_id"] = fp[fp.find("/")+1:-4]
    c["file_path"] = fp
    return (News.create(c), {
        "news_id": c["news_id"],
        "story"  : c["story"],
    })


def announcement_obj_creator(c, fp):
    c["announcement_id"] = fp.replace("/", "-")[:-4]
    c["file_path"] = fp
    return (Announcement.create(c), {
        "announcement_id": c["announcement_id"],
        "story"          : c["story"]
    })


######################
ptt_parse_loop = parse_loop_wrapper(ptt_parse_unit, ptt_obj_creator)
news_parse_loop = parse_loop_wrapper(news_parse_unit, news_obj_creator)
announcement_parse_loop = parse_loop_wrapper(announcement_parse_unit, announcement_obj_creator)


######################
def json_dumper(js_objs, ofilename):
    with open(ofilename, "w", encoding="utf-8") as f:
        json.dump(js_objs, f, ensure_ascii=False)
    return


def database_dumper(db_objs):
    session = DBSession()
    session.add_all(db_objs)
    session.commit()
    return


def run():
    config = [
        ("財訊", "./announcement-json/announcement-", announcement_parse_loop),
        # ("中時", "news", news_parse_loop),
        # ("ptt", "./ptt-json/ptt-", ptt_parse_loop)
    ]
    for (folder_name, json_name, looper) in config:
        for folder_name, db_objs, js_objs in looper(
            ifolder="/Users/Kevin/Temp/twquant/news/"+folder_name,
            # file_selector=lambda fln, fn: fln == "20181128" and fn.startswith("00117")
            file_selector=lambda fln, fn: True
        ):
            pass
            # database_dumper(db_objs)
            json_dumper(js_objs, json_name+folder_name+".json")

    return


if __name__ == '__main__':
    run()
