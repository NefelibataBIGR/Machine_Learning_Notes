import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def knn_demo():
    '''
    knn预测鸢尾花类别
    '''
    # 1、读取数据
    iris = load_iris()
    # print(iris)

    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=1)
    # print(type(x_train))

    # 3、特征工程：标准化
    trans = StandardScaler()
    x_train = trans.fit_transform(x_train) # 对x_train进行标准化
    x_test = trans.transform(x_test) # 对x_test也进行相同的标准化

    # 4、训练knn模型
    esti = KNeighborsClassifier(n_neighbors=3) # 实例化一个预估器
    esti.fit(x_train, y_train) # 训练

    # 5、评估模型
        # 方法1：
    y_pred = esti.predict(x_test) # 预测
    print("y_predict =", y_pred)
    print("accuracy_1 =", np.sum(y_pred == y_test)/sum(np.ones(y_test.shape))) # 计算准确率

        # 方法2：
    accuracy = esti.score(x_test, y_test) # 计算准确率
    print("accuracy_2 =", accuracy)

def knn_gscv():
    '''
    knn+网格搜索+交叉验证:预测鸢尾花类别
    '''
    # 1、读取数据
    iris = load_iris()
    # print(iris)

    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=1)
    # print(type(x_train))

    # 3、特征工程：标准化
    trans = StandardScaler()
    x_train = trans.fit_transform(x_train) # 对x_train进行标准化
    x_test = trans.transform(x_test) # 对x_test也进行相同的标准化

    # 4、训练knn模型
    esti = KNeighborsClassifier() # 实例化一个预估器

    para_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11]}
    esti = GridSearchCV(esti, param_grid=para_grid, cv=10) # 网格搜索+交叉验证

    esti.fit(x_train, y_train) # 训练

    # 5、评估模型
        # 方法1：
    y_pred = esti.predict(x_test) # 预测
    print("y_predict =", y_pred)
    print("accuracy_1 =", np.sum(y_pred == y_test)/sum(np.ones(y_test.shape))) # 计算准确率

        # 方法2：
    accuracy = esti.score(x_test, y_test) # 计算准确率
    print("accuracy_2 =", accuracy)

    # 最佳参数：best_params_
    print("最佳参数：\n", esti.best_params_)
    # 最佳结果：best_score_
    print("最佳结果：\n", esti.best_score_)
    # 最佳估计器：best_estimator_
    print("最佳估计器:\n", esti.best_estimator_)
    # 交叉验证结果：cv_results_
    print("交叉验证结果:\n", esti.cv_results_)

def naive_bayes():
    '''
    朴素贝叶斯算法，新闻分类
    '''
    # 获取数据集
    # 划分数据集
    # 特征工程：文本特征抽取（Tf-idf）
    # 朴素贝叶斯算法
    # 模型评估

    news = fetch_20newsgroups(subset="all") # 获取数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, random_state=11) # 划分数据集

    # 特征工程：文本特征抽取（Tf-idf）
    trans = TfidfVectorizer()
    x_train = trans.fit_transform(x_train)
    x_test = trans.transform(x_test)
    print("type of x_test:", type(x_test))

    # 朴素贝叶斯算法
    esti = MultinomialNB()
    esti.fit(x_train, y_train)
    
    # 模型评估
        # 方法1：
    y_pred = esti.predict(x_test) # 预测
    print("y_predict =", y_pred)
    accuracy_1 = np.sum(y_pred == y_test)/sum(np.ones(y_test.shape))
    print("accuracy_1 =", accuracy_1) # 计算准确率

        # 方法2：
    accuracy_2 = esti.score(x_test, y_test) # 计算准确率
    print("accuracy_2 =", accuracy_2)

def tree():
    '''
    决策树分类：鸢尾花数据集
    '''
    iris = load_iris() # 读取数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=11) # 划分数据集

    esti = DecisionTreeClassifier(criterion='entropy') # 实例化预估器
    esti.fit(x_train, y_train) # 训练

    # 可视化决策树
    export_graphviz(esti, out_file='my_iris_tree.dot', feature_names=iris.feature_names) # 加上feature_names能让树状图更容易看懂
    
    # 模型评估
        # 方法1：
    y_pred = esti.predict(x_test) # 预测
    print("y_predict =", y_pred)
    accuracy_1 = np.sum(y_pred == y_test)/sum(np.ones(y_test.shape))
    print("accuracy_1 =", accuracy_1) # 计算准确率

        # 方法2：
    accuracy_2 = esti.score(x_test, y_test) # 计算准确率
    print("accuracy_2 =", accuracy_2)

if __name__ == '__main__':
    # 1、knn预测鸢尾花类别
    # knn_demo()

    # 2、knn+网格搜索+交叉验证:预测鸢尾花类别
    # knn_gscv()

    # 3、朴素贝叶斯算法，新闻分类
    # naive_bayes()

    # 4、决策树分类：鸢尾花数据集
    tree()