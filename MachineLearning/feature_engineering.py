import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import jieba

def datasets_demo():
    '''
    sklearn数据集的使用
    '''
    # 获取数据集
    iris = load_iris()
    # print("iris dataset is like :\n", iris)
    # print("dataset descr:\n", iris['DESCR']) # 用键-值对查看
    # print("feature names:\n", iris.feature_names)
    # print("data:\n", iris.data, iris.data.shape)

    # 数据集的划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)
    print("The data of train dataset is like:\n", x_train)

    return None

def dict_demo():
    '''
    字典特征提取
    '''
    data = [{'city': '北京','temperature':100}, 
            {'city': '上海','temperature':60}, 
            {'city': '深圳','temperature':30}] # 数据要是字典的迭代器，可以列表包含字典
    
    # 1、实例化一个转换器类：
    trans = DictVectorizer(sparse=False) # one-hot编码需要sparse=False

    # 2、调用fit_transform()
    data_new = trans.fit_transform(data)

    print("new data:\n", data_new)
    print("feature names:\n", trans.get_feature_names_out())

    return None

def text_demo():
    '''
    英文文本特征提取
    '''
    text = ["life is short,i like like python", 
            "life is too long,i dislike python"]
    
    # 类似dict_demo()
    trans = CountVectorizer()
    text_new = trans.fit_transform(text)

    # print("new text:\n", text_new) # 返回sparse矩阵，不方便观察
    print("new text:\n", text_new.toarray())
    print("feature names:\n", trans.get_feature_names_out())

def text_Chinese_try():
    '''
    中文文本特征提取（尝试）
    '''
    text = ["我爱北京天安门", 
            "天安门上太阳升"]
    
    # 类似dict_demo()
    trans = CountVectorizer()
    text_new = trans.fit_transform(text)

    # print("new text:\n", text_new) # 返回sparse矩阵，不方便观察
    print("new text:\n", text_new.toarray())
    print("feature names:\n", trans.get_feature_names_out())

def Chinese_cut(text):
    '''
    传入的要是字符串，不是列表！
    用jieba剪切中文语句，自动分词
    '''
    text_new = ' '.join(jieba.cut(text))
    return text_new

def text_Chinese_demo():
    '''
    中文自动分词
    '''
    # 1、分词
    text = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # text = "一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。"

    text_new = []
    for sen in text:
        text_new.append(Chinese_cut(sen))
    
    # print("分词后：\n",text_new)
    
    # 2、特征提取
    # 类似dict_demo()
    trans = CountVectorizer()
    # trans = CountVectorizer(stop_words=["一种", "所以"]) # 可以去掉没啥意义的词

    text_final = trans.fit_transform(text_new)

    # print("new text:\n", text_new) # 返回sparse矩阵，不方便观察
    print("new text:\n", text_final.toarray())
    print("feature names:\n", trans.get_feature_names_out())

def keyword_demo():
    '''
    文本关键词特征提取
    '''
    # 1、分词
    text = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # text = "一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。"

    text_new = []
    for sen in text:
        text_new.append(Chinese_cut(sen))
    
    # print("分词后：\n",text_new)
    
    # 2、特征提取
    # 类似dict_demo()
    trans = TfidfVectorizer()
    # trans = TfidfVectorizer(stop_words=["一种", "所以"]) # 可以去掉没啥意义的词

    text_final = trans.fit_transform(text_new)

    # print("new text:\n", text_new) # 返回sparse矩阵，不方便观察
    print("new text:\n", text_final.toarray())
    print("feature names:\n", trans.get_feature_names_out())

def norm_demo():
    '''
    特征数据归一化
    '''
    data = pd.read_csv("./day1资料/02-代码/dating.txt")
    # print("data:\n", data.head())
    data_feat = data.iloc[:, :3] # 只索引所需列

    trans = MinMaxScaler() # 实例化一个转换器类!!!
    # trans = MinMaxScaler(feature_range=(0, 100))
    data_new = trans.fit_transform(data_feat)
    
    print("normalized data:\n",data_new)
    print("feature names:\n", trans.get_feature_names_out())

def stan_demo():
    '''
    特征数据标准化
    '''
    data = pd.read_csv("./day1资料/02-代码/dating.txt")
    # print("data:\n", data.head())
    data_feat = data.iloc[:, :3] # 只索引所需列

    trans = StandardScaler() # 实例化一个转换器类!!!
    # trans = StandardScaler(with_mean= , with_std= )
    data_new = trans.fit_transform(data_feat)
    
    print("standarded data:\n",data_new)
    print("feature names:\n", trans.get_feature_names_out())

def filter_demo():
    '''
    过滤式特征选择
    '''
    # 方差选择法：

    data = pd.read_csv("./day1资料/02-代码/factor_returns.csv")
    # print("data is:\n", data.head()) # 发现有字符串

    data_feat = data.iloc[:, 1:-2]
    # print("data_feat is:\n", data_feat.head())

    trans = VarianceThreshold(threshold=5) # threshold方差阈值默认为0
    data_new = trans.fit_transform(data_feat)
    # print(type(data_new)) # 返回的是ndarray类型，没有head()函数

    print("filtered data:\n", data_new)
    print("shape:\n", data_new.shape)

    # 相关系数法：
    
    r = pearsonr(data['pe_ratio'], data['pb_ratio'])
    print("correlation coefficient is:", r)

def PCA_demo():
    '''
    主成分分析(PCA)降维
    '''
    data = [[2,8,4,5], 
            [6,3,0,8], 
            [5,4,9,1]]
    data = np.array(data) # 这一步类型转换可以忽略
    # print(type(data))

    trans = PCA(n_components=0.8)
    # trans = PCA(n_components=3)
    data_new = trans.fit_transform(data)

    print("PCAed data:\n", data_new)

if __name__ == '__main__':
    
    # 1、sklearn数据集的使用
    # datasets_demo()

    # 2、字典特征提取
    # dict_demo()

    # 3、英文文本特征提取
    # text_demo()

    # 4、中文文本特征提取（尝试）
    # text_Chinese_try()

    # 5、中文自动分词
    # text_Chinese_demo()

    # 6、文本关键词特征提取
    # keyword_demo()

    # 7、特征数据归一化
    # norm_demo()

    # 8、特征数据标准化
    # stan_demo()

    # 9、过滤式特征选择
    # filter_demo()

    # 10、主成分分析(PCA)降维
    PCA_demo()