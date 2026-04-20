import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

def linear_LR():
    '''
    正规方程：预测加州房价

    流程：
    1. 获取数据集
    2. 划分数据集
    3. 特征工程
        无量纲化 - 标准化
    4. 预估器流程
    5. 模型评估（调参）
    '''
    house = fetch_california_housing() # 获取数据集（注意需要魔法）
    print("特征数量", house.data.shape)
    x_train, x_test, y_train, y_test = train_test_split(house.data, house.target, random_state=11) # 划分数据集
    # 需要比较正规方程和随机梯度下降哪个好，所以随机数种子设置相同

    # 标准化
    trans = StandardScaler()
    x_train = trans.fit_transform(x_train)
    x_test = trans.transform(x_test)

    # 预估器流程
    esti = LinearRegression()
    esti.fit(x_train, y_train)

    # 得出模型
    print("linear_LR权重系数：\n", esti.coef_)
    print("linear_LR偏置：\n", esti.intercept_)

    # 评估模型
    y_pred = esti.predict(x_test)
    print("linear_LR预测房价：\n", y_pred)
    
    error = mean_squared_error(y_test, y_pred)
    print("linear_LR均方误差：\n", error)

def linear_SGDR():
    '''
    随机梯度下降：预测加州房价

    流程：
    1. 获取数据集
    2. 划分数据集
    3. 特征工程
        无量纲化 - 标准化
    4. 预估器流程
    5. 模型评估（调参）
    '''
    house = fetch_california_housing() # 获取数据集（注意需要魔法）
    # print("特征数量", house.data.shape)
    x_train, x_test, y_train, y_test = train_test_split(house.data, house.target, random_state=11) # 划分数据集，注意随机数种子要相同
    # 需要比较正规方程和随机梯度下降哪个好，所以随机数种子设置相同

    # 标准化
    trans = StandardScaler()
    x_train = trans.fit_transform(x_train)
    x_test = trans.transform(x_test)

    # 预估器流程
    esti = SGDRegressor()
    esti.fit(x_train, y_train)

    # 得出模型
    print("linear_SGDR权重系数：\n", esti.coef_)
    print("linear_SGDR偏置：\n", esti.intercept_)

    # 评估模型
    y_pred = esti.predict(x_test)
    print("linear_SGDR预测房价：\n", y_pred)
    
    error = mean_squared_error(y_test, y_pred)
    print("linear_SGDR均方误差：\n", error)

def linear_Ridge():
    '''
    岭回归：预测加州房价

    流程：
    1. 获取数据集
    2. 划分数据集
    3. 特征工程
        无量纲化 - 标准化
    4. 预估器流程
    5. 模型评估（调参）
    '''
    house = fetch_california_housing() # 获取数据集（注意需要魔法）
    # print("特征数量", house.data.shape)
    x_train, x_test, y_train, y_test = train_test_split(house.data, house.target, random_state=11) # 划分数据集，注意随机数种子要相同
    # 需要比较正规方程和随机梯度下降哪个好，所以随机数种子设置相同

    # 标准化
    trans = StandardScaler()
    x_train = trans.fit_transform(x_train)
    x_test = trans.transform(x_test)

    # 预估器流程
    esti = Ridge(random_state=11) # 可以调参优化，调α等参数
    esti.fit(x_train, y_train)

    # 得出模型
    print("linear_Ridge权重系数：\n", esti.coef_)
    print("linear_Ridge偏置：\n", esti.intercept_)

    # 评估模型：均方误差评估
    y_pred = esti.predict(x_test)
    print("linear_Ridge预测房价：\n", y_pred)
    
    error = mean_squared_error(y_test, y_pred) # 均方误差评估
    print("linear_Ridge均方误差：\n", error)

if __name__ == '__main__':
    # 1、正规方程：预测加州房价
    linear_LR()

    # 2、随机梯度下降：预测加州房价
    linear_SGDR()

    # 3、岭回归：预测加州房价
    linear_Ridge()