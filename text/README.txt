text 文件夹中主要是处理 事理图谱构建和事件抽取的内容

文件基本介绍

核心代码部分:
concrete-graph 第三章 用于制作具体事理图谱(具体事理图谱的官方译名用的是specific)
    *** paratactic-sentence-graph 3.1 流水句图谱构建
    ** causality-extractor 和 compound-extractor 用模板抽取因果关系和并列关系
    ** compress-clause 用于压缩短句(但其实实际中没有用到,仅做学习使用,里面的算法是树形递归算法)
    *** clause-similarity 3.2 用词袋模型计算短句相似度并给出短句的cluster, 并且融合图谱

    * concrete-graph 入口文件,读取数据库并调用如上代码
    * evaluate 表格7 用于评估具体事理图谱
    * script 过程中一些脚本函数,可以忽略
    * conf 用于设定各个环节的输出内容,可以忽略

abstract-graph 第四章 用于制作抽象事理图谱

    * word-cluster 计算词的聚类,用了Word2vec和tfidf,但是效果不好,最后的同义词表还是我人工建的
    ** extract-node-feature 利用词典计算节点特征
    ** node-event-mapping 利用节点特征和事件特征,找出节点最可能属于的抽象事件类型
    * feature-uils 一些工具包用来抽取特征

    * abstract-graph 入口文件,读取数据库并调用如上代码
    * evaluate 表格7 用于评估具体事理图谱
    * script 过程中一些脚本函数,可以忽略
    * conf 用于设定各个环节的输出内容,可以忽略

event-extraction 第四章 用于在预测股价时从新闻标题抽取事件

核心词典部分
concrete-vocab 具体事理图谱中用的词典
    1. coord-conjs 并列连词
    2. psych-verbs 心理情感动词
    3. stoplist 停用词表

abstract-vocab 抽象事理图谱中用的词典
    1. general-subject和verb是我人工做的,通过代码转为*_cluster的形式,实际代码中用的是后者
    2. negative和positive并没有用在模型中,但是对于本数据集是有效的
    3. geographical-cluster也没有使用,但是有效,基本分为:全球/大陆/台湾/日本/美国/欧洲/其他等类
    4. tfidf和word2vec仅为构建词典测试时使用,正式模型中没有使用

ltp-model 基础自然语言处理用到的模型

其他代码
news2company 用于确定新闻和公司的映射关系,会将结果输出到数据库的 stock_code_self 字段
plot_price2story 用于展示新闻和股价的趋势图(前期我用来做数据可视化,但现在不一定可用)
preprocess 将原始新闻的文本进行分行分段,或者分点,形成表格等(日后处理其他类型的announcement可以参考这个做预处理)
xml_parse 将xml解析为json和数据库元数据(基本以后用不到了,仅做学习使用)