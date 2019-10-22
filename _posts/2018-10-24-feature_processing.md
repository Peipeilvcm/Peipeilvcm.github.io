---
layout: post
title: feature_processing
description: 特征工程
tag: DataScience

---

![特征工程](/images/posts/DataScience/feature_processing.jpg "特征工程")

## 1 数据数据预处理

特征当前问题

- 不同量纲，不能放在一起比较，无量纲化
- 定量特征信息冗余，二值化，区间划分
- 定性特征在某些模型中不能直接使用，需转换为定量特征。方法：OneHotEncoder, 哑编码
- 缺失值，插值，丢弃
- 信息利用率低，定量变量多项式化

### 1.1 无量纲化

#### 1.1.1 标准化

前提：特征值服从正太分布
转化为均值为0，方差为1的标准正太分布

```
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
std_scaler.fit(X_train)
X_train_std = std_scaler.transform(X_train)
X_test_std = std_scaler.transform(X_test)
```

#### 1.1.2 区间缩放法

```python
from sklearn.preprocessing import MinMaxScaler, RobustScaler
#RobustScaler中位数和四分位数，会忽略异常值
#所有特征位于0~1之间,要求未知数据大小范围不超过训练数据
```

#### 1.1.3 归一化

标准化处理矩阵的列数据，各特征值转换至同一量纲下
归一化处理行数据，使样本向量在点乘运算下或其它和函数计算相似性时有统一标准，转换为“单位向量”。

L1,L2归一化 :
$$
x'=\dfrac {x}{\sqrt {\sum ^{m}_{h=1}x^{2}_{m}}}
$$

```python
from sklearn.preprocessing import Normalizer
```

### 1.2 定量特征二值化

设定阈值，大于threshold为1，小与为0

```python
from sklearn.preprocessing import Binarizer
Binarizer(threshold=3).fit_transform(X_train)
```

### 1.3 定性特征哑编码

```python
from sklearn.preprocessing import OneHotEncoder
df = pd.get_dummies(df,['col1_name','col3_name'])
```

### 1.4 异常值处理

计算缺失值可填充为mean,median,most_frequent,插值等

```python
#sklearn方法
from sklearn.impute import SimpleImputer

#pandas方法
isnull,duplicated
dropna()，drop()，drop_duplicated()
fillna()，新属性，集中值，边界值；插值 interpolate()
```

### 1.5 数据变换

方法：多项式，指数函数，对数函数

```python
from sklearn.preprocessing import PolynomialFeatures
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
FunctionTransformer(log1p).fit_transform(X_train)
```

## 2 特征选择

- 特征是否发散，若方差接近于0，可舍弃
- 特征与目标相关性
- feature_selection库

### 2.1 Filter过滤法

按照发散性或相关性对个特征评分，设定阀值，选择特征

#### 2.1.1 方差选择法

计算各特征方差，根据阀值，选择方差大于阀值的特征。

```python
from sklearn.feature_selection import VarianceThreshold
VarianceThreshold(threshold=3).fit_transform(X_train)
```

#### 2.1.2 相关系数、卡方效验

计算各特征与目标的相关系数，取前K个特征，相关系数有pearson， spearman，chi2

```python
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
X_new = SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
或
X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
```

卡方效验，定性自变量对定性因变量的相关性

#### 2.1.3 互信息法

评价定性自变量对定性因变量的相关性

```python
from sklearn.metrics import mutual_info_score
mutual_info_score(label,x)

from sklearn.feature_selection import SelectKBest
from minepy import MINE
#MINE的设计不是函数式的，定义mic方法将其为函数式
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return m.mic(), 0.5
X_new = SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(X_train, y_train)
```

### 2.2 Wrapper 包装法

#### 2.2.1 递归特征消除RFE

使用一个基模型进行多轮训练，每轮训练后选出最好的（或最差的）特征放到一边，在剩余特征上重复此过程，直到所有特征都遍历了。

REF的稳定性很大程度取决于底层使用哪个模型。如采用普通回归，没有经过正则化，是不稳定的。若采用Ridge，L_2 范数惩罚项；Lasso，L1范数惩罚项；是稳定的。

```python
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
estimator = SVR(kernel="linear")
selector = RFE(estimator, 5, step=1)
selector = selector.fit(X, y)
print(selector.support_)
```

#### 2.2.2 稳定性选择

基于二次抽样和选择算法相结合，选择算法可以是回归、SVM等。主要思想：是在不同的数据子集和特征子集上运行特征选择算法，不断的重复，最终汇总特征选择结果。可以统计某个特征被认为是重要特征的频率（被选为重要特征的次数除以它所在的子集被测试的次数）

### 2.3 Embedded嵌入法

#### 2.3.1 基于惩罚项

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(X, y)
```

L1惩罚项保留多个对目标值具有同等相关性特征中的一个。没选到不一定不重要，结合L2惩罚项优化。若一个特征在L1中的权值为1，选择在L2中权值差别不大且在L1中权值为0的特征构成同类集合，将这一集合中的特征平分L1中的权值。

#### 2.3.2 基于树模型

GBDT也可用来作为基模型进行特征选择

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
SelectFromModel(GradientBoostingClassifier()).fit_transform(X, y)
```

## 3 降维

![降维方法](/images/posts/DataScience/dimension_reduction.png)

PCA主成分分析，映射后样本最发散，特征提取，白化，将主成分缩放至同一尺度。

SVD奇异值矩阵分解

FA因子分析

LDA主成分分析(有监督)，投影变换后，同一标注距离小，不同的距离大。n_components <= n_classes - 1。

NMF非负矩阵分解，用于特征提取，每个数据点写成一些分量的求和，希望分量和系数都大于或等于0，只能应用于每个特征都是非负的数据，可以用来识别组成合成数据的原始分量。分成的分量地位平等。（主要参数，提取分量个数）

t-SNE流形学习用于二维可视化，尽可能保持数据点之间的距离，让原始特征空间中距离较近的点更加靠近，原始较远的更加远离。不允许变换新数据，不能用于测试集。监督学习很少使用。参数perplexity,early_exaggeration,一般默认就很好，改参数作用小。

## 4 常见模型

### 4.1 回归



### 4.2 分类

KNN:k越小模型越复杂，近似误差小，容易过拟合。

SVM:核宽度gamma和正则化参数C，C越小，强调系数越趋于0，正则化越强，泛化能力越好，C越大越容易过拟合。

### 4.3 聚类

Kmeans:把每个数据点分配给最近的簇中心，每个簇中心设置为分配的所有数据点的平均值。依赖于初始化，矢量量化可以用比输入维度更多的族来对数据编码。缺点：依赖于随机初始化，对簇形状约束性强，要制定簇的个数。

凝聚聚类：首先每个点是一个簇，合并两个最相似的簇，直到满足到指定簇个数。AgglomerativeCluster。

DBSCAN:不需设置簇个数，核心样本，噪声。参数min_samples和eps，一个数据点eps的距离内至少有min_samples点，它就是核心样本。不允许对新的测试数据进行预测。

K-means,圆去圈类别；混合高斯模型，椭圆圈类别；谱聚类spectral_clustering，弯曲数据的聚类；

Kmeans对所有数据点进行划分，DBSCAN有噪声概念，凝聚聚类可以提供可能的层次结构。

聚类评估：聚类个数与损失函数L直接相互妥协，Elbow mothod手肘法(假设聚类个数小于真实类别数，聚类结果误差平方下降快，大于时下降慢)，升级为更严谨的sihouette analysis。调整Rand指数ARI，归一化互信息NMI，轮廓系数silhouette coeffcient描述簇的紧致度(实际效果不好)。还有基于鲁棒性的聚类指标，sklearn还未实现。



## 一些常见问题

#### a. 不平衡数据

使用AUC评价。更改类别权重，权重为占比倒数，使其损失函数中分类错误的惩罚更大class_weight=balanced。还有重新抽样方法。参考Learning from Imbalanced Data

#### b. 模型持久化

pcikle.dump可以把训练好的模型保存到磁盘上，pickle.load可以读取。

#### c. 定性变量与定量变量

定性变量，onehotencoder，再取一个或多个基准（选择比列最大的为基准类），去掉非显著类。当类别太多，区间计数法，特征哈希方案。

定量变量，卡方检验，取最适合区间分段分类，用线性回归或逻辑斯特回归，考查模型参数的显著性，参数估计值与估计值的标准差。用决策树，能综合考虑多个数值型变量，变为类别型变量。

#### d. 共线性问题

使模型参数不稳定，估计值标准差变大。

多重共线性检验：

- 两个定量变量，相关系数(Pearson与Spearman)和相应的P-value。
- 两个定性变量，卡方统计量和相应P-value。
- 定量变量A与定性变量B，one-way ANOVA中技术指标。
- 对于多个变量，联合假设检验和方差膨胀因子。

数据共线性方法：

- 增加数据量
- 去掉共线性的不重要变量
- 数据降维，主成分分析PCA
- 加入惩罚项
- 忽略共线性问题

结构化共线性(x与x^2):

- 将变量中心置为0，变量归一化
- 与上面相同

#### 内生性问题

被预测量与自变量互为因果。

来源：遗漏变量，度量误差，自变量与被预测量的同时性。使模型参数不准确。

解决方法，工具变量。

## 参考

[使用sklearn做单机特征工程](https://www.cnblogs.com/jasonfreak/p/5448385.html)
