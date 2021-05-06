# coding=utf-8
from database_model.text_model import *
from config import MYSQL_URI
import pymysql
pymysql.install_as_MySQLdb()
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(MYSQL_URI)
DBSession = sessionmaker(bind=engine)


def init_db():
    """
        生成数据库表
    :return:
    """
    Base.metadata.create_all(engine)


def drop_db():
    """
        删除数据库表
    :return:
    """
    Base.metadata.drop_all(engine)


if __name__ == '__main__':
    init_db()
    # drop_db()