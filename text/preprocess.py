import re

import zhconv

import utils as ut


def line_return_preprocess(story):
    """
    这个函数主要处理分行问题,分行有三种情况:表格/句末分行/分点分行
    :param story:
    :return:
    """

    # 1. 判断是否为表格, 如果是表格则转为简体中文后原格式输出
    def is_table(s, ln=5, sn=10):
        lines = s.split("\n")
        spaces = ut.map_list(lambda line: len(line.strip().split(" ")), lines)
        return len(ut.filter_list(lambda space_num: space_num > sn, spaces)) > ln

    if is_table(story):
        return ut.map_list(lambda s: zhconv.convert(s, "zh-cn"),
                           ut.map_list(lambda s: s.strip(), story.split("\n")))

    # 2. 如果不是表格先进行句末分行
    sents = ut.map_list(lambda s: s.strip()+"。", filter(lambda s: len(s.strip()), story.split("。\n")))

    # 判断是否为分点的算法,输入为一个句子片段,输出是否匹配
    def is_point_wrapper():
        dsetB = ['零', '一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十']
        dset = dsetB+[str(i) for i in range(10)]

        sdset = ''.join(dset)

        sp = ['\(['+sdset+']{1,2}\)',
              '['+sdset+']{1,2}[、.]{1}']
        pattern = re.compile(u'|'.join(sp))

        return lambda s: pattern.match(s)

    is_point = is_point_wrapper()

    # 2.1 遍历一个分句,将其中需要删掉的句内分行删除,将其中的分点分行保留
    def split_contents(sent, k=5):
        i = 0
        while len(sent) > i:
            if sent[i] != "\n" or is_point(sent[(i+1):(i+k)]):
                i += 1
            else:
                sent = sent[:i]+sent[i+1:]
        return sent.split("\n")

    # 2.2 根据清理分行\n之后的结果进行重新分行并打平列表
    sents = ut.flatten(ut.map_list(lambda sent: split_contents(sent), sents))

    return ut.map_list(lambda s: zhconv.convert(s, "zh-cn"), sents)


if __name__ == '__main__':
    test1 = "公告本公司董事會決議召開股東會相關事宜。\n1.董事會決議日期:107/02/27\n2.股東會召開日期:107/05/23\n3.股東會召開地點:高雄市永安區永工一路5號(" \
            "本公司)\n4.召集事由一、報告事項:\n(1)106年度營業報告。\n(2)106年度監察人查核報告。\n(3)106年度員工酬勞及董監酬勞分配情形報告。\n(4)106年度對外背書保證執行情形報告。\n(" \
            "5)106年度大陸投資概況。\n5.召集事由二、承認事項:\n(1)106年度營業報告書及財務報告案。\n(" \
            "2)106年度盈餘分配案。\n6.召集事由三、討論事項:修訂本公司「取得或處分資產處理程序」部分條文案。\n7.召集事由四、選舉事項:無\n8.召集事由五、其他議案:無\n9.召集事由六、臨時動議:無\n10" \
            ".停止過戶起始日期:107/03/25\n11.停止過戶截止日期:107/05/23\n12.其他應敘明事項:\n受理股東提案公告、作業流程及審查標準：\n依公司法第172條之1" \
            "規定，持有已發行股份總數百分之一以上股份之股東，得以書面\n向公司提出股東常會議案，但以一項並以三百字為限。本公司擬訂於107年3月16日起\n至107年3月26" \
            "日止受理股東就本次股東常會之提案，凡有意提案之股東務請於\n107年3月26日17時前送達並敘明聯絡人及聯絡方式，以利董事會審查及回覆審查結果。\n" \
            "請於信封封面上加註『股東會提案函件』字樣，以掛號函件寄送。\n受理處所：勝一化工(股)公司(地址：高雄市永安區永工一路5號)\n有下列情事之一，股東所提議案，董事會得不列為議案：\n(" \
            "一)該議案非股東會所得決議者。\n(二)提案股東於停止過戶日時，持股未達百分之一者。\n(三)該議案於公告受理期間外提出者。\n"

    lines = line_return_preprocess(test1)
    for l in lines:
        print(l)
