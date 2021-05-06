from database_model.base import Base

from sqlalchemy import BigInteger, Column, DateTime, Enum, Index, SmallInteger, String, Text, \
    UniqueConstraint, and_, exists, not_, Integer
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from datetime import datetime


class News(Base):
    __tablename__ = 'news'

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    news_id = Column(String(32), nullable=False, index=True)
    file_path = Column(String(255))

    provider = Column(String(32))
    datetime = Column(DateTime)
    headline = Column(String(255))
    author = Column(String(32))
    cat = Column(String(32))
    url = Column(MEDIUMTEXT)
    picture_url = Column(MEDIUMTEXT)
    keywords = Column(String(255))

    @staticmethod
    def create(news_dict):
        news = News(
            news_id=news_dict["news_id"],
            file_path=news_dict["file_path"],

            provider=news_dict["providor"],
            datetime=datetime.strptime(news_dict["datetime"], "%Y-%m-%d %H:%M:%S"),
            headline=news_dict["headline"],
            author=news_dict["author"],
            cat=news_dict["cat"],
            url=news_dict["sourceurl"],
            picture_url=news_dict.get("picture"),
            keywords=news_dict["keywords"],
        )
        return news

    UniqueConstraint('news_id', name='uk_news_id')


class Ptt(Base):
    __tablename__ = 'ptt'

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    ptt_id = Column(String(32), nullable=False, index=True)
    file_path = Column(String(255))

    board = Column(String(32))
    post_time = Column(DateTime)
    title = Column(String(255))
    author = Column(String(32))

    ip = Column(String(32))
    url = Column(MEDIUMTEXT)

    like_num = Column(Integer)
    unlike_num = Column(Integer)
    null_num = Column(Integer)

    @staticmethod
    def create(ptt_dict):
        ptt = Ptt(
            ptt_id=ptt_dict["ptt_id"],
            file_path=ptt_dict["file_path"],
            post_time=datetime.strptime(ptt_dict["postTime"], "%Y/%m/%d %H:%M:%S"),
            board=ptt_dict["board"],
            title=ptt_dict["title"],
            author=ptt_dict["author"],
            ip=ptt_dict["ip"],
            url=ptt_dict["url"],
            like_num=ptt_dict["like_num"],
            unlike_num=ptt_dict["unlike_num"],
            null_num=ptt_dict["null_num"],

        )
        return ptt

    UniqueConstraint('ptt_id', name='uk_ptt_id')


class Announcement(Base):
    __tablename__ = 'announcement'

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    announcement_id = Column(String(32), nullable=False, index=True)
    file_path = Column(String(255))

    provider = Column(String(32))
    datetime = Column(DateTime)
    source = Column(String(32))

    headline = Column(String(255))
    priority = Column(String(16))
    type = Column(String(64))
    type_org = Column(String(16))
    mid_code = Column(String(64))
    style = Column(String(16))
    country_area = Column(String(64))
    industry = Column(String(64))
    stock_code = Column(Text)
    stock_code_self = Column(Text)
    image = Column(String(64))
    effect_level = Column(String(64))
    main_stock = Column(String(64))
    author = Column(String(64))
    tag = Column(String(64))
    top = Column(String(64))


    @staticmethod
    def create(announcement_dict):
        announcement = Announcement(
            announcement_id=announcement_dict["announcement_id"],
            file_path=announcement_dict["file_path"],
            provider=announcement_dict["provider"],
            source=announcement_dict["source"],
            datetime=datetime.strptime(announcement_dict["datetime"], "%Y%m%d%H%M%S"),
            headline=announcement_dict["headline"],
            priority=announcement_dict["priority"],
            type=announcement_dict["type"],
            type_org=announcement_dict["type_org"],
            mid_code=announcement_dict["mid_code"],
            style=announcement_dict["style"],
            country_area=announcement_dict["countryarea"],
            industry=announcement_dict["industry"],
            stock_code=announcement_dict["stockcode"],
            image=announcement_dict["image"],
            effect_level=announcement_dict["effectLevel"],
            main_stock=announcement_dict["mainstock"],
            author=announcement_dict["author"]
        )
        return announcement

    UniqueConstraint('announcement_id', name='uk_announcement_id')

