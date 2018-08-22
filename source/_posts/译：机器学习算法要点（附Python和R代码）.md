---
title: 译：机器学习算法要点（附Python和R代码）
categories: Python + R
tags: [机器学习,线性回归,逻辑回归,决策树,支持向量机,朴素贝叶斯,最近邻（KNN）,K-均值,随机森林,降维算法,梯度下降算法,GBM,XGBoost,LightGBM,CatBoost]
---
最近一直忙于琐事，无力抽身学习，现抽出两天点滴时间，粗略翻译了一篇机器学习算法要点（附Python和R代码），以便需要之时查阅。此篇文章并未涉及深度学习、迁移学习、对抗学习等内容，待后续补充。

“谷歌的自动驾驶汽车和机器人受到了很多媒体的关注，但该公司真正的未来在于机器学习，这是一种让计算机变得更智能、更个性化的技术。”
                                                ——埃里克.施密特(谷歌董事会主席)
<!--more-->                                               
我们可能生活在人类历史上最具决定性的时期。计算从大型主机到pc再到云计算的时代。但它的定义并不是发生了什么，而是未来几年将会发生什么。

对于像我这样的人来说，这段时期令人兴奋的是工具和技术的民主化，这是随着计算机技术的发展而来的。今天，作为一名数据科学家，我可以用复杂的算法构建数据处理机器，每小时几美元。但是，做到这些并不容易！如果我没有没日没夜的付出。

## 谁将从这份指南获益匪浅
创建这个指南的目的是为了简化世界各地有抱负的数据科学家和机器学习爱好者的学习路径。通过本指南，我将使您能够处理机器学习问题并从经验中获益。我对各种机器学习算法以及R和Python代码进行了高水平的理解，并运行它们。这些应该足够让你的手忙碌了。

![image](http://wx2.sinaimg.cn/large/e621a9abgy1flb2wio330j20dr09eq4k.jpg)

我故意跳过了这些技术背后的统计知识，因为您在开始时不需要了解它们。所以，如果你在寻找对这些算法的统计理解，你应该去别处看看。但是，如果你想要装备自己来开始建造机器学习项目，你完全可以信任它。
## 总的来说，有三种类型的机器学习算法
1. 监督学习

它是如何工作的:这个算法由一个变量/因变量(或依赖变量)组成，这个变量是由一个给定的预测集(独立变量)来预测的。使用这组变量，我们生成一个将输入映射到所需输出的函数。培训过程将继续，直到模型达到训练数据所需的精确程度。监督学习的例子:线性回归、决策树、随机森林、最近邻（KNN)、逻辑回归等。

2. 无监督学习

它是如何工作的:在这个算法中，我们没有任何目标或结果变量来预测/估计。它适用于不同组的聚类，广泛用于细分不同群体，以进行特定的干预。无监督学习的例子:关联算法，k-均值。

3. 强化学习

它是如何工作的:使用这个算法，机器被训练来做出具体的决定。它的工作原理是这样的:机器被暴露在一个环境中，在这个环境中，它不断地通过试错来训练自己。这台机器从过去的经验中学习，并试图获取最好的知识以做出准确的决策。强化学习的例子:马尔科夫决策过程。
## 一般机器学习算法列表
- 线性回归
- 逻辑回归
- 决策树
- 支持向量机
- 朴素贝叶斯
- 最近邻（KNN）
- K-均值
- 随机森林
- 降维算法
- 梯度下降算法
    1. GBM
    2. XGBoost
    3. LightGBM
    4. CatBoost

## 1.线性回归
它用于估计基于连续变量(s)的实际值(房子的成本，电话的数量，总销售额等)。在这里，我们通过拟合最佳直线，建立了独立变量和依赖变量之间的关系。这条最佳拟合线被称为回归线，由线性方程Y=aX+b表示。

理解线性回归最好的方法就是重温童年的经历。比如说，你让一个五年级的孩子通过体重递增方式排列他的同学，而不问他们的具体重量！你认为孩子会做什么?他/她可能会(在视觉上分析)身高和身材，并利用这些可见的参数来排列他们。这是现实生活中的线性回归！这个孩子实际上已经知道了身高与体重之间的关联，这看起来就像上面的等式。

在这个方程中:

Y -因变量

a -斜率

X -独立变量

b -截距

这些系数a和b是基于最小化数据点和回归线之间距离的平方差的。

请看下面的例子。这里我们已经确定了线性方程y=0.2811x+13.9的最佳拟合线。利用这个方程，我们可以计算出一个人的身高。

![image](http://wx4.sinaimg.cn/mw690/e621a9abgy1flb2x6zbpxj20d807vjr9.jpg)

线性回归主要有两种类型:一元线性回归和多元线性回归。一元线性回归具有一个独立变量的特征。并且，多元线性回归(如名字所示)具有多个(2个以上)独立变量的特征。在寻找最佳拟合线时，你可以选择一个多项式或曲线回归。这些被称为多项式或曲线线性回归。

Python Code

```python
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import linear_model
#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays
x_train=input_variables_values_training_datasets
y_train=target_variables_values_training_datasets
x_test=input_variables_values_test_datasets
# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)
#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
#Predict Output
predicted= linear.predict(x_test)
```
R Code

```r
#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays
x_train <- input_variables_values_training_datasets
y_train <- target_variables_values_training_datasets
x_test <- input_variables_values_test_datasets
x <- cbind(x_train,y_train)
# Train the model using the training sets and check score
linear <- lm(y_train ~ ., data = x)
summary(linear)
#Predict Output
predicted= predict(linear,x_test) 
```
## 2.逻辑回归
别被它的名字搞糊涂了！它是一种分类而不是回归算法。它用于估计离散值(二进制值，如0/1、yes/no、true/false)，基于给定的独立变量(s)。简单地说，它通过将数据匹配到logit函数来预测事件发生的可能性。因此，它也被称为logit回归。因为它预测了概率，它的输出值在0到1之间(如预期的那样)。

让我们再通过一个简单的例子来理解这个问题。

假设你的朋友给了你出了一道题。只有两种结果——要么你解决了，要么你没有。现在想象一下，你被给予了各种各样的智力测验/小测验，试着去理解你擅长的科目。这个研究的结果是这样的-如果你有一个基于十年级的三角测量法，你有70%的可能会解决这个问题。另一方面，如果这是第五次历史问题，得到答案的概率只有30%。这就是逻辑回归所提供的。

在数学计算中，结果的对数概率被建模为预测变量的线性组合。

```
odds= p/ (1-p) = probability of event occurrence / probability of not event occurrence
ln(odds) = ln(p/(1-p))
logit(p) = ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk
```
上面，p是我们感兴趣的特征的概率。它选择参数来最大化观察样本值的可能性，而不是最小化平方误差的总和(就像普通的回归一样)。

现在，你可能会问，为什么要写一个日志呢?为了简单起见，我们假设这是复制一个阶跃函数的最好的数学方法之一。我可以详细介绍，但这将超出本文的目的。

![image](http://wx1.sinaimg.cn/mw690/e621a9abgy1flbgv1oc3ij20et0bawer.jpg)

Python Code

```python
#Import Library
from sklearn.linear_model import LogisticRegression
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create logistic regression object
model = LogisticRegression()
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)
#Predict Output
predicted= model.predict(x_test)
```
R Code

```r
x <- cbind(x_train,y_train)
# Train the model using the training sets and check score
logistic <- glm(y_train ~ ., data = x,family='binomial')
summary(logistic)
#Predict Output
predicted= predict(logistic,x_test)
```
另外，如果想提高改进模型，有许多不同的方法可以尝试:
- 包括相互作用项
- 删减特征
- 正则化技术
- 使用非线性模型
## 3.决策树
这是我最喜欢的算法之一，我经常使用它。它是一种被监督的学习算法，主要用于分类问题。令人惊讶的是，它适用于分类和连续相关变量。使用这个算法，我们可以把种群分成两个或更多的同类集合。它基于最重要的属性/独立变量，以使其尽可能地成为不同的组。要了解更多细节，您可以阅读有关决策树的建模实现过程。

![image](http://wx1.sinaimg.cn/mw690/e621a9abgy1flbo75c9onj20g30bumyh.jpg)

在上面的图片中，你可以看到，根据不同的属性，人们被划分为四个不同的组，以确定“他们是否会出去玩”。为了将人口分成不同的不同的群体，它使用了各种各样的技术，比如基尼系数、信息增益、卡方、熵。

要理解决策树的工作原理，最好的方法是玩Jezzball——这是微软的一款经典游戏(下图)。基本上，你有一间有移动墙的房间，你需要创建墙壁，这样没有球的最大区域就会被清理。

![image](http://wx3.sinaimg.cn/mw690/e621a9abgy1flboei37ijj208204uwei.jpg)

所以，每次你用一面墙把房间分成两部分，你就会在同一个房间里创造两种不同的人群。决策树以非常相似的方式工作，将一个种群划分为不同的组。

Python Code

```python
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import tree
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create tree object 
model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)
```
R Code

```r
library(rpart)
x <- cbind(x_train,y_train)
# grow tree 
fit <- rpart(y_train ~ ., data = x,method="class")
summary(fit)
#Predict Output 
predicted= predict(fit,x_test)
```
## 4. SVM (支持向量机)
这是一种分类方法。在这个算法中，我们将每个数据项绘制成n维空间中的一个点(其中n是您拥有的特性的数量)，每个特性的值都是一个特定坐标的值。

例如，如果我们只有两个特征，比如身高和头发的长度，我们首先会在二维空间中绘制这两个变量，其中每个点都有两个坐标(这些坐标被称为支持向量)。

![image](http://wx4.sinaimg.cn/mw690/e621a9abgy1flboor8ne8j20nm0fz76m.jpg)

现在，我们将找到一些在两种不同类别的数据之间分割的线。这条线是这样的，这两个簇两个最近点之间是最远的。

![image](http://wx2.sinaimg.cn/mw690/e621a9abgy1flboowe1szj20nm0g2gpo.jpg)

在上面的示例中，将数据划分为两个不同的分组的那条线是黑线，因为这两个最接近的点离黑线最远。这条线是我们的分类器。然后，根据测试数据位置在这条线的哪一边，对新数据进行分类。

如果把这个算法看作是在n维空间中玩JezzBall。游戏的微调是:

- 你可以在任何角度画线/平面(而不是像传统游戏那样水平或垂直)。

- 游戏的目的是在不同的房间中分离不同颜色的球。

- 球是固定不动的。

Python Code
```python
#Import Library
from sklearn import svm
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.svc() # there is various option associated with it, this is simple for classification. You can refer link, for mo# re detail.
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)
```
R Code
```r
library(e1071)
x <- cbind(x_train,y_train)
# Fitting model
fit <-svm(y_train ~ ., data = x)
summary(fit)
#Predict Output 
predicted= predict(fit,x_test)
```
## 5. 朴素贝叶斯
这是一种基于贝叶斯定理的分类技术，前提基于预测者之间的独立性假设。简单地讲，朴素贝叶斯的分类器假设一个类的特定特性与其他任何特性的存在无关。例如，如果水果是红色的，圆形的，直径大约3英寸，就可以认为它是苹果。即使这些特征相互依赖，或者其他特征的存在，一个朴素贝叶斯分类器也会考虑所有这些特性各自独立地为这个水果是苹果的概率可能性做出的贡献。

简单的贝叶斯模型很容易构建，对于非常大的数据集尤其有用。除了简洁性，朴素贝叶斯也被认为比高度复杂的分类方法表现出色。

贝叶斯定理提供了一种由P(c)、P(x)和P(x|c)中计算出概率P(c|x)的方法。请看下面的等式:

![image](http://wx1.sinaimg.cn/mw690/e621a9abgy1flc14kmymtj20ct07c744.jpg)

这里，
- P(c|x)是分类(目标)的后验概率(属性)。
- P(c)是分类的先验概率。
- P(x|c)是预测给定分类的概率可能性。
- P(x)是预测的先验概率。

例子:让我们用一个例子来理解它。下面我有一组天气预报和相应的目标变量“Play”。现在，我们需要根据天气情况对玩家是否会出去玩进行分类。让我们按照下面的步骤来执行它。

步骤1:将数据集转换为频率表

第2步:通过寻找可能性的概率来创建可能性表，例如:阴天概率=0.29，而玩的概率是0.64。

![image](http://wx2.sinaimg.cn/mw690/e621a9abgy1flc1ezajkzj20nm08m0us.jpg)

第三步:用简单的贝叶斯方程来计算每个类的后验概率。具有最高后验概率的类是预测的结果。

问题:如果天气晴朗，球员会出去玩，这句话是正确的吗?

我们可以用上面讨论的方法来解它，所以 P(Yes | Sunny) = P( Sunny | Yes) * P(Yes) / P (Sunny)

这里P (Sunny |Yes) = 3/9 = 0.33, P(Sunny) = 5/14 = 0.36, P( Yes)= 9/14 = 0.64

现在，P (Yes | Sunny) = 0.33 * 0.64 / 0.36 = 0.60，这有更高的概率。

Naive Bayes使用了一种类似的方法来预测不同类型的不同属性的可能性。该算法主要用于文本分类和有多个类的问题。

Python Code
```python
#Import Library
from sklearn.naive_bayes import GaussianNB
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object model = GaussianNB() # there is other distribution for multinomial classes like Bernoulli Naive Bayes
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
```
R Code
```r
library(e1071)
x <- cbind(x_train,y_train)
# Fitting model
fit <-naiveBayes(y_train ~ ., data = x)
summary(fit)
#Predict Output 
predicted= predict(fit,x_test)
```
## 6. KNN (K-最近邻)
它可以用于分类和回归问题。然而，它更广泛地应用于工业中的分类问题。K最近邻是一个简单的算法，它可以存储所有可用的情况，并通过它的K个邻居的多数投票来对新情况进行分类。被分配到一个类的实例在它的K最近的邻居中也是最常见的，它是由一个距离函数测量的。

这些距离函数可以是欧氏、曼哈顿、闵可夫斯基和海明距离。第3个函数用于连续函数，第4个(海明)用于分类变量。如果K=1,那么，这个实例被简单地分配给它最近的邻居。很多时候，在使用KNN建模时，对于K的选择往往是一个挑战。

![image](http://wx3.sinaimg.cn/mw690/e621a9abgy1flc1xehlh9j20jv08p742.jpg)

KNN很容易被映射到我们的真实生活中。如果你想了解一个你没有任何信息的人，你可能会想了解他的亲密朋友和他的圈子，并由此获得他/她的信息！

在选择KNN之前要考虑的事情:
- KNN在计算上代价是昂贵的
- 变量应该被规范化否则更高范围的变量会导致偏差
- 在使用KNN之前进行预处理十分重要，比如离群值，消除噪音

Python Code
```python
#Import Library
from sklearn.neighbors import KNeighborsClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create KNeighbors classifier object model 
KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicte
```
R Code
```r
library(knn)
x <- cbind(x_train,y_train)
# Fitting model
fit <-knn(y_train ~ ., data = x,k=5)
summary(fit)
#Predict Output 
predicted= predict(fit,x_test)
```
## 7. K-均值
它是一种不受监督的算法，解决了聚类的问题。它的过程遵循一种简单而容易的方法，通过一定数量的簇(假设k个簇)对给定的数据集进行分类。簇中的数据点是同类的，异构的，对等的。

还记得从墨水污渍中找出形状吗?k均值与其有点类似。你看一下这个形状，然后扩散开来，看看有多少不同的簇出现！

![image](http://wx1.sinaimg.cn/mw690/e621a9abgy1flc2c9mzbbj207w08cq2y.jpg)

k-均值聚簇方法:
- k-均值为每个中心点挑选出K个点。
- 每个数据点形成一个具有最接近的中心点的簇，即k个簇。
- 根据现有的簇成员查找每个簇的中心点。这里有新的中心点。
- 当我们有新的中心点时，重复第2步和第3步。从新的中心找到每个数据点的最近距离，并与新的k个簇联系起来。重复这个过程直到收敛发生，即中心点不会改变。

如何确定K的值:

在k-均值中，我们有这样的簇，每个簇都有自己的中心点。中心点和一个簇中的数据点之间的平方和构成了该集群的平方值的总和。此外，当添加了所有簇的平方总和时，它就变成了簇解决方案的平方总和。

我们知道，随着集群的数量增加，这个值一直在减少但如果你画出结果，你会发现，距离的平方和k的值会急剧下降，之后会更慢。我们可以以此找到最优簇个数K。

![image](http://wx2.sinaimg.cn/mw690/e621a9abgy1flc2o0yhv0j20nm0bxab6.jpg)

Python Code
```python
#Import Library
from sklearn.cluster import KMeans
#Assumed you have, X (attributes) for training data set and x_test(attributes) of test_dataset
# Create KNeighbors classifier object model 
k_means = KMeans(n_clusters=3, random_state=0)
# Train the model using the training sets and check score
model.fit(X)
#Predict Output
predicted= model.predict(x_test)
```
R Code
```r
library(cluster)
fit <- kmeans(X, 3) # 3 cluster solution
```
## 8. 随机森林
随机森林是一组决策树的标术语。在随机森林中，我们收集了决策树(也就是所谓的“森林”)。为了对基于属性的新对象进行分类，每棵树都给出一个分类，我们说这个树给类进行“投票”。森林选择了拥有最多选票的分类。

每棵树都是种下并生长的:
- 如果训练集的实例数是N，则随机可替换抽取这N个实例的样本。这个样本将作为树生长的训练集。
- 如果有M个输入变量，那么在每个节点上都指定了一个m值的M变量，m变量是在M中中随机选择的，在这些m中最好的分割是去用来分割节点的。在森林生长过程中，m的值保持不变。
- 每棵树都在尽可能大的范围内生长。没有修剪。

Python Code
```python
#Import Library
from sklearn.ensemble import RandomForestClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Random Forest object
model= RandomForestClassifier()
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
```
R Code
```r
library(randomForest)
x <- cbind(x_train,y_train)
# Fitting model
fit <- randomForest(Species ~ ., x,ntree=500)
summary(fit)
#Predict Output 
predicted= predict(fit,x_test)
```
## 9. 降维算法
在过去的4-5年中，在每个可能的阶段，数据收集都呈指数级增长。企业/政府机构/研究机构不仅提供了新的信息来源，而且还在详细地收集数据。

例如:电子商务公司正在获取更多关于客户的细节信息，比如他们的人口统计数据、网络爬行历史、他们喜欢或不喜欢的东西、购买历史记录、反馈信息等等，这些都比你最近的杂货店老板更能给他们个性化的关注。

作为一名数据科学家，我们提供的数据也包含许多特性，这对于构建良好的健壮模型来说是很好的，但也有一个挑战。你如何从1000或2000变量中识别出高度显著的变量(s)?在这种情况下，降维算法可以帮助我们，与决策树、随机森林、主成分分析、因子分析等其他算法相结合，以关联矩阵、缺失值比例等为基础进行识别。

Python Code
```python
#Import Library
from sklearn import decomposition
#Assumed you have training and test data set as train and test
# Create PCA obeject pca= decomposition.PCA(n_components=k) #default value of k =min(n_sample, n_features)
# For Factor analysis
#fa= decomposition.FactorAnalysis()
# Reduced the dimension of training dataset using PCA
train_reduced = pca.fit_transform(train)
#Reduced the dimension of test dataset
test_reduced = pca.transform(test)
```
R Code
```r
library(stats)
pca <- princomp(train, cor = TRUE)
train_reduced  <- predict(pca,train)
test_reduced  <- predict(pca,test)
```
## 10. 梯度下降算法
>  10.1. GBM

GBM是一种增强算法，当我们以高预测能力处理大量数据时，它是一种增强算法。增强实际上是一套学习算法的集合，它结合了几个基本估计量的预测，以提高对单一估计量的鲁棒性。它将多个弱或平均的预测因子结合构建到一个强的预测器中。这些增强算法在诸如Kaggle、AV Hackathon、CrowdAnalytix等数据科学竞赛中都很有效。

Python Code
```python
#Import Library
from sklearn.ensemble import GradientBoostingClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Gradient Boosting Classifier object
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
```
R Code
```r
library(caret)
x <- cbind(x_train,y_train)
# Fitting model
fitControl <- trainControl( method = "repeatedcv", number = 4, repeats = 4)
fit <- train(y ~ ., data = x, method = "gbm", trControl = fitControl,verbose = FALSE)
predicted= predict(fit,x_test,type= "prob")[,2] 
```
> 10.2. XGBoost

另一种经典的梯度下降算法，它被认为是在某些Kaggle比赛中获胜与失败之间的决定性选择。

XGBoost具有非常高的预测能力，这使它在预测模型中成为最佳的选择，因为它既具有线性模型又有树学习算法，使得算法比现有的梯度下降技术快了10倍。

它可以支持的功能，包括回归、分类和排名。

XGBoost最有趣的一点是它也被称为规则化的增强技术。这有助于减少过度拟合的建模，并对诸如Scala、Java、R、Python、Julia和C++等一系列语言提供了大量支持。

它也支持分布式大规模训练，包括GCE、AWS、Azure和Yarn 集群。XGBoost也可以与Spark、Flink和其他云数据系统集成在一起，在每次迭代的过程中都建立了交叉验证。

Python Code
```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = dataset[:,0:10]
Y = dataset[:,10:]
seed = 1

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

model = XGBClassifier()

model.fit(X_train, y_train)

#Make predictions for test data
y_pred = model.predict(X_test)
```
R Code
```r
require(caret)

x <- cbind(x_train,y_train)

# Fitting model

TrainControl <- trainControl( method = "repeatedcv", number = 10, repeats = 4)

model<- train(y ~ ., data = x, method = "xgbLinear", trControl = TrainControl,verbose = FALSE)

OR 

model<- train(y ~ ., data = x, method = "xgbTree", trControl = TrainControl,verbose = FALSE)

predicted <- predict(model, x_test)
```
> 10.3. LightGBM

LightGBM是一个使用基于树的学习算法的梯度增强框架。它的设计是分布式的和高效的，有以下优点:
- 更快的训练速度和更高的效率
- 降低内存使用
- 更好的精度
- 并行和GPU学习支持
- 能够处理大规模数据

该框架是一种基于决策树算法的快速和高性能梯度增强，用于排序、分类和许多其他机器学习任务。它是在微软的分布式机器学习工具包项目下开发的。

由于LightGBM是基于决策树算法的，它将树叶和最适合的树分割开，而其他的提升算法将树的深度和水平都分割开，而不是叶节点。因此，当LightGBM在相同的叶子生长时，叶节点的智慧算法能够比水平的算法减少更多的损失，从而产生更好的精度，而这在现有的增强算法中几乎是不可能实现的。

而且，它的速度惊人的快，因此用了“光”这个词。

Python Code
```python
data = np.random.rand(500, 10) # 500 entities, each contains 10 features
label = np.random.randint(2, size=500) # binary target

train_data = lgb.Dataset(data, label=label)
test_data = train_data.create_valid('test.svm')

param = {'num_leaves':31, 'num_trees':100, 'objective':'binary'}
param['metric'] = 'auc'

num_round = 10
bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])

bst.save_model('model.txt')

# 7 entities, each contains 10 features
data = np.random.rand(7, 10)
ypred = bst.predict(data)
```
R Code
```r
library(RLightGBM)
data(example.binary)
#Parameters

num_iterations <- 100
config <- list(objective = "binary",  metric="binary_logloss,auc", learning_rate = 0.1, num_leaves = 63, tree_learner = "serial", feature_fraction = 0.8, bagging_freq = 5, bagging_fraction = 0.8, min_data_in_leaf = 50, min_sum_hessian_in_leaf = 5.0)

#Create data handle and booster
handle.data <- lgbm.data.create(x)

lgbm.data.setField(handle.data, "label", y)

handle.booster <- lgbm.booster.create(handle.data, lapply(config, as.character))

#Train for num_iterations iterations and eval every 5 steps

lgbm.booster.train(handle.booster, num_iterations, 5)

#Predict
pred <- lgbm.booster.predict(handle.booster, x.test)

#Test accuracy
sum(y.test == (y.pred > 0.5)) / length(y.test)

#Save model (can be loaded again via lgbm.booster.load(filename))
lgbm.booster.save(handle.booster, filename = "/tmp/model.txt")
```
如果您熟悉R中的Caret包，下面是实现LightGBM的另一种方式。
```r
require(caret)
require(RLightGBM)
data(iris)

model <-caretModel.LGBM()

fit <- train(Species ~ ., data = iris, method=model, verbosity = 0)
print(fit)
y.pred <- predict(fit, iris[,1:4])

library(Matrix)
model.sparse <- caretModel.LGBM.sparse()

#Generate a sparse matrix
mat <- Matrix(as.matrix(iris[,1:4]), sparse = T)
fit <- train(data.frame(idx = 1:nrow(iris)), iris$Species, method = model.sparse, matrix = mat, verbosity = 0)
print(fit)
```
> 10.4. Catboost

CatBoost是最近开源的一款来自Yandex的机器学习算法。它可以很容易地与谷歌的TensorFlow和苹果的Core ML等深度学习框架集成。

关于CatBoost的最好的地方是它不需要像其他ML模型那样进行大量的数据进行训练，并且可以处理各种数据格式，也不会影响它的鲁棒性。

在执行算法之前，请确保您已对缺失的数据进行了处理。

Catboost可以自动处理分类变量，而不显示类型转换错误，这有助于您集中精力优化您的模型，而不是处理一些琐碎的错误。

Python Code
```python
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor

#Read training and testing files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Imputing missing values for both train and test
train.fillna(-999, inplace=True)
test.fillna(-999,inplace=True)

#Creating a training set for modeling and validation set to check model performance
X = train.drop(['Item_Outlet_Sales'], axis=1)
y = train.Item_Outlet_Sales

from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)
categorical_features_indices = np.where(X.dtypes != np.float)[0]

#importing library and building model
from catboost import CatBoostRegressormodel=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')

model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)

submission = pd.DataFrame()

submission['Item_Identifier'] = test['Item_Identifier']
submission['Outlet_Identifier'] = test['Outlet_Identifier']
submission['Item_Outlet_Sales'] = model.predict(test)
```
R Code
```r
set.seed(1)

require(titanic)

require(caret)

require(catboost)

tt <- titanic::titanic_train[complete.cases(titanic::titanic_train),]

data <- as.data.frame(as.matrix(tt), stringsAsFactors = TRUE)

drop_columns = c("PassengerId", "Survived", "Name", "Ticket", "Cabin")

x <- data[,!(names(data) %in% drop_columns)]y <- data[,c("Survived")]

fit_control <- trainControl(method = "cv", number = 4,classProbs = TRUE)

grid <- expand.grid(depth = c(4, 6, 8),learning_rate = 0.1,iterations = 100, l2_leaf_reg = 1e-3,            rsm = 0.95, border_count = 64)

report <- train(x, as.factor(make.names(y)),method = catboost.caret,verbose = TRUE, preProc = NULL,tuneGrid = grid, trControl = fit_control)

print(report)

importance <- varImp(report, scale = FALSE)

print(importance)
```
## 写在最后
现在，我敢肯定，你会有一种大体如何使用机器学习算法的想法。我写这篇文章并提供R和Python代码的唯一目的就是让你马上开始。如果你想要精通机器学习，那就马上开始吧。解决问题，不断加强对过程的物理理解，应用这些代码，并发现乐趣！

译自：https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/