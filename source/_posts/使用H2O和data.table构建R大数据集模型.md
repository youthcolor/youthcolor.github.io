---
title: 使用H2O和data.table构建R大数据集模型
categories: R
tags: [大数据,H2O]
---
## 引文
上周，我写了一篇关于data.table包的介绍性文章。它的目的是为您提供一个良好的开端，并熟悉它的独特和简短的语法。下一步是关注建模，我们将在今天的文章中做建模。

有了data.table，您不再需要担心您的机器没有足够的RAM。至少，在面对大型数据集时，我曾经认为自己是一个瘫痪的R用户。但不会有这种担心了，再次感谢[Matt Dowle](https://www.linkedin.com/in/mattdowle)的做出的贡献。
<!--more-->
上周，我收到一封邮件说:“好的,我明白了。data.table使我们能够进行数据的探索和操作。但是，模型构建呢?我的RAM为8G。像随机森林(ntrees=1000)这样的算法需要花费太长时间来运行80万行数据集。”

我确信有很多的R用户被困在类似的情况下。为了克服这一艰难的障碍，我决定编写这篇文章，它演示了如何使用两个最强大的包，即H2O和data.table。

为了达到实用性的理解，我从一个[实践问题](http://datahack.analyticsvidhya.com/contest/black-friday)中获取了数据，并尝试使用4种不同的机器学习算法(H2O)和特性工程(使用data.table)来提高分数。所以，在排行榜上准备好从排行第154位到排行第25位的旅程。

## 目录
1. H2O简介
2. 为什么它如此快
3. 实战开始
4. 使用data.table和ggplot进行数据探索
5. 使用data.table处理数据
6. 使用H2O构建模型
    - 回归
    - 随机森林
    - GBM
    - 深度学习

注意:将本文作为使用data.table和H2O进行数据模型构建的初学者指南。我还没有详细解释这些算法。相反，我们关注的焦点是如何使用H2O来实现这些算法。别担心，资源和链接都会提供的。
## H2O简介
[H2O](http://www.h2o.ai/)是一个开源机器学习平台，企业可以在大型数据集(无采样需求)上建立模型，并实现准确的预测。它的速度非常快，可伸缩，并且很容易实现。

简而言之，它们为企业提供了一个GUI驱动的平台，以便进行更快的数据计算。目前，他们的平台支持高级和基本的机器学习算法，如深度学习、提升、装袋、朴素贝叶斯、主成分分析、时间序列、k-均值、广义线性模型。

此外，H2O还为R、Python、Spark和Hadoop用户提供了api，让我们这样的人可以使用它在各个层次上构建模型。它可以自由使用，并加快计算速度。
## 为什么它如此快
H2O有一个干净而清晰的特性，可以直接将工具(R或Python)与您的计算机的CPU连接起来。这样我们就可以获得更多的内存，处理更快速的计算工具的能力。这将允许计算以100%的CPU容量进行(如下所示)。它还可以在云平台的集群上进行计算。

此外，它还使用内存中的压缩来处理大型数据集，即使有一个小的集群。它还提供了并行分布式式网络训练的支持。
## 实战开始
数据集:我从黑色星期五的实践问题中获取数据。数据集有两部分:训练集和测试集。训练数据集包含550068个观测数据。测试数据集包含233599个观察结果。下载数据并阅读问题声明:[点击这里](https://datahack.analyticsvidhya.com/contest/black-friday/)。需要登录。

理想情况下，模型构建的第一步是生成假设。在您阅读了问题陈述之后，但是没有看到数据，这一步是有必要的。

因为，这个指南并不是为了演示所有的预测建模步骤，所以我把它留给您。这里有一个很好的资源来更新你的基础知识:[假设生成向导](https://www.analyticsvidhya.com/blog/2015/09/hypothesis-testing-explained/)。如果你做了这一步，也许你最终会创造出比我更好的模型。尽你最大的努力。

H2O官网在R中安装H2O方法：
```r
# The following two commands remove any previously installed H2O packages for R.
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
# Next, we download packages that H2O depends on.
pkgs <- c("RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}
# Now we download, install and initialize the H2O package for R.
install.packages("h2o", type="source", repos="http://h2o-release.s3.amazonaws.com/h2o/rel-wheeler/1/R")
# Finally, let's load H2O and start up an H2O cluster
library(h2o)
h2o.init()
```
开始在R中加载数据
```r
path <- "D:\\用户目录\\下载"
setwd(path)
#install and load the package
install.packages("data.table")
library(data.table)
#load data using fread
train <- fread("train.csv", stringsAsFactors = T)
test <- fread("test.csv", stringsAsFactors = T)
```
在几秒钟内，fread将数据加载到r中，速度很快。参数stringsAsFactors确保将字符向量转换为因子。让我们快速检查一下数据集。
```r
dim(train)
[1] 550068     12
#No. of rows and columns in Test
dim(test)
[1] 233599     11
str(train)
Classes ‘data.table’ and 'data.frame':	550068 obs. of  12 variables:
 $ User_ID                   : int  1000001 1000001 1000001 1000001 1000002 1000003 1000004 1000004 1000004 1000005 ...
 $ Product_ID                : Factor w/ 3631 levels "P00000142","P00000242",..: 673 2377 853 829 2735 1832 1746 3321 3605 2632 ...
 $ Gender                    : Factor w/ 2 levels "F","M": 1 1 1 1 2 2 2 2 2 2 ...
 $ Age                       : Factor w/ 7 levels "0-17","18-25",..: 1 1 1 1 7 3 5 5 5 3 ...
 $ Occupation                : int  10 10 10 10 16 15 7 7 7 20 ...
 $ City_Category             : Factor w/ 3 levels "A","B","C": 1 1 1 1 3 1 2 2 2 1 ...
 $ Stay_In_Current_City_Years: Factor w/ 5 levels "0","1","2","3",..: 3 3 3 3 5 4 3 3 3 2 ...
 $ Marital_Status            : int  0 0 0 0 0 0 1 1 1 1 ...
 $ Product_Category_1        : int  3 1 12 12 8 1 1 1 1 8 ...
 $ Product_Category_2        : int  NA 6 NA 14 NA 2 8 15 16 NA ...
 $ Product_Category_3        : int  NA 14 NA NA NA NA 17 NA NA NA ...
 $ Purchase                  : int  8370 15200 1422 1057 7969 15227 19215 15854 15686 7871 ...
 - attr(*, ".internal.selfref")=<externalptr>
```
我们看到了什么?我们看到了12个变量，其中2个看起来有那么多的NAs。如果你读过问题描述和数据信息，我们会看到购买是依赖变量，剩下11个是独立变量。

在购买变量(连续)的性质上，我们可以推断这是一个回归问题。尽管比赛已经结束了，但是我们仍然可以检查我们的分数，并评估我们的表现有多好。让我们第一次提交看看结果如何。

有了所有的数据点，我们可以用均值来做第一组预测。这是因为，平均预测会给我们一个很好的预测误差的近似值。将此作为基线预测，我们的模型不会比这更糟。
```r
#first prediction using mean
sub_mean <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = mean(train$Purchase))
write.csv(sub_mean, file = "first_sub.csv", row.names = F)
```
这个比较简单。现在，我将上传结果文件并检查我的分数和等级。别忘了转换csv。上传之前的压缩格式。你可以在竞争页面上上传和检查你的解决方案。

![image](https://www.analyticsvidhya.com/wp-content/uploads/2016/05/21-1024x264.png)

我们的平均预测给了我们一个均值的平方误差4982.3199。但是，它有多好呢?让我们来看看我在排行榜上的排名。

![image](https://www.analyticsvidhya.com/wp-content/uploads/2016/05/22-1024x443.png)

谢天谢地，我不是最后一个。所以，平均预测得到了154/162。让我们改进这一分数，并试图提升排行榜的排名。

在开始使用单变量分析之前，让我们快速地总结一下文件(训练集和测试集)和解释，如果存在任何差异的话。

仔细看(在结尾)，你看到他们的输出有什么不同吗?事实上,我发现了一个。如果你仔细比较产品类别1、产品类别2和产品类别的测试和训练数据，你就会在最大的价值上存在差异。产品分类的最大值是20，而其他的是18。这些额外的类别似乎是噪声。请注意这一点。我们需要移除它们。

让我们把数据集合并起来，我使用了来自data.table的rbindlist函数，因为它比rbind函数速度快。
```r
#combine data set
test[,Purchase := mean(train$Purchase)]
c <- list(train, test)
combin <- rbindlist(c)
```
在上面的代码中，我们首先在测试集中添加了购买变量，这样两个数据集都有相同数量的列。现在，我们来做一些数据探索。
## 使用data.table和ggplot进行数据探索
在本节中，我们将做一些单变量和变量对分析，并尝试理解给定变量之间的关系。让我们先从单变量开始。
```r
#analyzing gender variable
combin[,prop.table(table(Gender))]
Gender
        F         M 
0.2470896 0.7529104 
#Age Variable
combin[,prop.table(table(Age))]
Age
      0-17      18-25      26-35      36-45      46-50      51-55        55+ 
0.02722330 0.18113944 0.39942348 0.19998801 0.08329814 0.06990724 0.03902040 
#City Category Variable
combin[,prop.table(table(City_Category))]
City_Category
        A         B         C 
0.2682823 0.4207642 0.3109535 
#Stay in Current Years Variable
combin[,prop.table(table(Stay_In_Current_City_Years))]
Stay_In_Current_City_Years
        0         1         2         3        4+ 
0.1348991 0.3527327 0.1855724 0.1728132 0.1539825 
#unique values in ID variables
length(unique(combin$Product_ID))
[1] 3677
length(unique(combin$User_ID))
[1] 5891
#missing values
colSums(is.na(combin))
                   User_ID                 Product_ID                     Gender                        Age 
                         0                          0                          0                          0 
                Occupation              City_Category Stay_In_Current_City_Years             Marital_Status 
                         0                          0                          0                          0 
        Product_Category_1         Product_Category_2         Product_Category_3                   Purchase 
                         0                     245982                     545809                          0
```
以下是我们可以从单变量分析中得出的推论:
1. 我们需要将性别变量编码为0和1。
2. 我们还需要分箱重新编码这个年龄变量。
3. 由于在city category中有三个级别，所以我们可以进行一种热编码。
4. 目前的“4+”水平需要重新评估。
5. 数据集不包含所有惟一的id。这给了我们足够的特性工程的提示。
6. 只有两个变量有缺失值。事实上，很多缺失值，可能是一种隐藏的趋势。我们需要以不同的方式对待他们。

我们已经从单变量分析中得到了足够的提示。让我们快速地进行二元分析。通过添加更多参数，您可以使这些图表看起来很漂亮。这里有一个快速的学习[ggplots](https://www.analyticsvidhya.com/blog/2016/03/questions-ggplot2-package-r/)的指南。
```r
library(ggplot2)
#Age vs Gender
ggplot(combin, aes(Age, fill = Gender)) + geom_bar()
```
![image](https://www.analyticsvidhya.com/wp-content/uploads/2016/05/23.png)
```r
#Age vs City_Category
ggplot(combin, aes(Age, fill = City_Category)) + geom_bar()
```
![image](https://www.analyticsvidhya.com/wp-content/uploads/2016/05/24.png)

我们还可以创建交叉表来分析分类变量。为了创建交叉表，我们将使用创建全面交叉表的包gmodel。
```r
library(gmodels)
CrossTable(combin$Occupation, combin$City_Category)
```
有了这两个变量，您就可以获得一个长时间的全面交叉表。类似地，您可以在最后分析其他变量。我们的二元分析并没有给我们提供很多可操作的见解。不管怎样，我们现在要进行数据处理。
## 使用data.table处理数据
在这一部分中，我们将创建新的变量，重新评估现有变量，并处理缺失的值。简而言之，我们将为建模阶段准备好数据。
让我们从缺少的值开始。我们看到产品类别和产品类别有很多缺失值。对我来说，这暗示了一个隐藏的趋势，它可以通过创建一个新的变量来映射。因此，我们将创建一个新的变量，将NAs作为1和非NAs，在变量product类别2和product类别3中捕获。
```r
#create a new variable for missing values
combin[,Product_Category_2_NA := ifelse(sapply(combin$Product_Category_2, is.na) ==    TRUE,1,0)]
combin[,Product_Category_3_NA := ifelse(sapply(combin$Product_Category_3, is.na) ==  TRUE,1,0)]
```
现在让我们用任意的数字来估算缺失的值。让我们取-999
```r
#impute missing values
combin[,Product_Category_2 := ifelse(is.na(Product_Category_2) == TRUE, "-999",  Product_Category_2)]
combin[,Product_Category_3 := ifelse(is.na(Product_Category_3) == TRUE, "-999",  Product_Category_3)]
```
在继续进行特性工程之前，我们将从单变量分析中重新评估变量的值。
```r
#set column level
levels(combin$Stay_In_Current_City_Years)[levels(combin$Stay_In_Current_City_Years) ==  "4+"] <- "4"
#recoding age groups
levels(combin$Age)[levels(combin$Age) == "0-17"] <- 0
levels(combin$Age)[levels(combin$Age) == "18-25"] <- 1
levels(combin$Age)[levels(combin$Age) == "26-35"] <- 2
levels(combin$Age)[levels(combin$Age) == "36-45"] <- 3
levels(combin$Age)[levels(combin$Age) == "46-50"] <- 4
levels(combin$Age)[levels(combin$Age) == "51-55"] <- 5
levels(combin$Age)[levels(combin$Age) == "55+"] <- 6
#convert age to numeric
combin$Age <- as.numeric(combin$Age)
#convert Gender into numeric
combin[, Gender := as.numeric(as.factor(Gender)) - 1]
```
为了建模目的，将因子变量转换为数值或整数是明智的。

现在让我们一步一步来，创建更多新的变量a.k.a特性工程。

在单变量分析中，我们发现ID变量的惟一值与数据集中的总观察值相比要小一些，这意味着用户ID或productid必须多次出现在这个数据集中。

让我们创建一个新变量来捕获这些ID变量的计数。较高的用户数量表明，某个特定用户已经多次购买产品。高产品数量表明，一款产品已经被多次购买，这表明它的受欢迎程度。
```r
#User Count
combin[, User_Count := .N, by = User_ID]
#Product Count
combin[, Product_Count := .N, by = Product_ID]
```
同样，我们可以计算一个产品的平均购买价格。因为，购买价格越低，就越有可能被买到，反之亦然。类似地，我们可以创建另一个变量，该变量将用户的平均购买价格映射到用户的购买价格(平均)。
```r
#Mean Purchase of Product
combin[, Mean_Purchase_Product := mean(Purchase), by = Product_ID]
#Mean Purchase of User
combin[, Mean_Purchase_User := mean(Purchase), by = User_ID]
```
现在，我们只剩下一个关于city_category变量的热编码。这可以用dummies库来完成。
```r
library(dummies)
combin <- dummy.data.frame(combin, names = c("City_Category"), sep = "_")
```
在继续进行建模阶段之前，让我们检查一下变量的数据类型，并在必要时进行必要的更改。
```r
#check classes of all variables
sapply(combin, class)
#converting Product Category 2 & 3
combin$Product_Category_2 <- as.integer(combin$Product_Category_2)
combin$Product_Category_3 <- as.integer(combin$Product_Category_3)
```
## 使用H2O建模
在这一节中，我们将探讨在H2O中不同机器学习算法的力量。我们将构建带有回归、随机森林、GBM和深度学习的模型。

确保你不会像一个黑盒子那样使用这些算法。最好知道它们是如何工作的。这将帮助您理解构建这些模型所使用的参数。以下是学习这些算法的一些有用的资源:
1. 回归
2. 随机森林，GBM
3. 深度学习

但是,先做重要的事。让我们把数据集分成测试集和训练集。
```r
#Divide into train and test
c.train <- combin[1:nrow(train),]
c.test <- combin[-(1:nrow(train)),]
```
正如在开始时所发现的，在火车上的变量产品类别有一些噪声。让我们通过在产品类别1-18中选择所有行来删除它，从而删除类别为19和20的水平。
```r
c.train <- c.train[c.train$Product_Category_1 <= 18,]
```
现在，我们的数据集已经准备好进行建模了。
```r
localH2O <- h2o.init(nthreads = -1)
```
这个命令告诉H2O要使用机器上的所有cpu，这是推荐的。对于较大的数据集(比如1,000,000行)，H2O建议在具有高内存的服务器上运行集群，以获得最佳性能。一旦实例成功启动，您还可以使用下面的代码查看他的状态:
```r
h2o.init()
```
现在让我们将数据从R转移到H2O实例。它可以用as来完成H2O的命令。
```r
#data to h2o cluster
train.h2o <- as.h2o(c.train)
test.h2o <- as.h2o(c.test)
```
使用列索引，我们需要识别在建模中使用的变量:
```r
#check column index number
colnames(train.h2o)
 [1] "User_ID"                    "Product_ID"                 "Gender"                    
 [4] "Age"                        "Occupation"                 "City_Category_A"           
 [7] "City_Category_B"            "City_Category_C"            "Stay_In_Current_City_Years"
[10] "Marital_Status"             "Product_Category_1"         "Product_Category_2"        
[13] "Product_Category_3"         "Purchase"                   "Product_Category_2_NA"     
[16] "Product_Category_3_NA"      "User_Count"                 "Product_Count"             
[19] "Mean_Purchase_Product"      "Mean_Purchase_User"
```
让我们从多元线性回归模型开始。
### H2O中的多元线性回归
```r
regression.model <- h2o.glm( y = y.dep, x = x.indep, training_frame = train.h2o, family = "gaussian")
  |==================================================================================================| 100%
h2o.performance(regression.model)
H2ORegressionMetrics: glm
** Reported on training data. **
MSE:  16710563
RMSE:  4087.856
MAE:  3219.644
RMSLE:  0.5782911
Mean Residual Deviance :  16710563
R^2 :  0.3261543
Null Deviance :1.353804e+13
Null D.o.F. :545914
Residual Deviance :9.122547e+12
Residual D.o.F. :545898
AIC :10628689
```
H2O的GLM算法适用于各种类型的回归，如lasso、bridge、逻辑、线性等。用户只需要对family参数进行相应的修改。例如:要做逻辑回归，你可以改写成family=“binomial”。

因此，在我们打印了模型结果之后，我们看到，回归给出了一个可怜的R值，即0.326。它的意思是，依赖变量中只有32.6%的变量是由独立变量来解释的，而剩下的是无法解释的。这表明，回归模型无法捕获非线性关系。

出于好奇，让我们看看这个模型的预测。它会比平均预测更糟糕吗?
```r
#make predictions
predict.reg <- as.data.frame(h2o.predict(regression.model, test.h2o))
sub_reg <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase =  predict.reg$predict)
write.csv(sub_reg, file = "sub_reg.csv", row.names = F)
```
让我们上传解决方案文件(压缩格式)，检查我们是否有一些改进。

![image](https://www.analyticsvidhya.com/wp-content/uploads/2016/05/25-1024x304.png)

哇!我们的预测分数有所提高。我们从4982.31开始，随着回归，我们比以前的分数有了进步。在排行榜上，这份报告让我排到了第129位。

![image](https://www.analyticsvidhya.com/wp-content/uploads/2016/05/26-1024x141.png)

如果我们选择一种能很好地映射非线性关系的算法，我们就能做得很好。随机森林是我们的下一个赌注。
### H2O中的随机森林
```r
#Random Forest
system.time(
  rforest.model <- h2o.randomForest(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 1000, mtries = 3, max_depth = 4, seed = 1122)
)
```
在1000棵树的情况下，随机森林模型大约需要104秒的运行时间。它以100%的CPU容量运行，可以在任务管理器中看到。
```r
h2o.performance(rforest.model)
H2ORegressionMetrics: drf
** Reported on training data. **
** Metrics reported on Out-Of-Bag training samples **
MSE:  10414919
RMSE:  3227.215
MAE:  2486.118
RMSLE:  0.5007453
Mean Residual Deviance :  10414919
#check variable importance
h2o.varimp(rforest.model)
Variable Importances: 
                     variable     relative_importance scaled_importance percentage
1       Mean_Purchase_Product 2720452686381056.000000          1.000000   0.573797
2          Product_Category_1 1005997304840192.000000          0.369790   0.212185
3               Product_Count  252741091852288.000000          0.092904   0.053308
4          Product_Category_3  231408274505728.000000          0.085062   0.048809
5       Product_Category_3_NA  194243133964288.000000          0.071401   0.040970
6          Mean_Purchase_User  174858721820672.000000          0.064276   0.036881
7          Product_Category_2   84932466573312.000000          0.031220   0.017914
8       Product_Category_2_NA   54471002423296.000000          0.020023   0.011489
9                  User_Count   12314694647808.000000          0.004527   0.002597
10            City_Category_C    5007590031360.000000          0.001841   0.001056
11                     Gender    2175469223936.000000          0.000800   0.000459
12            City_Category_A    1162100736000.000000          0.000427   0.000245
13                        Age     613729370112.000000          0.000226   0.000129
14                 Occupation     478127718400.000000          0.000176   0.000101
15            City_Category_B     234770481152.000000          0.000086   0.000050
16 Stay_In_Current_City_Years      32139771904.000000          0.000012   0.000007
17             Marital_Status      17185155072.000000          0.000006   0.000004
```
让我们通过做预测来检查这个模型的排行榜。你认为我们的分数会提高吗?不过，我还是报点希望。
```r
#making predictions on unseen data
system.time(predict.rforest <- as.data.frame(h2o.predict(rforest.model, test.h2o)))
  |==================================================================================================| 100%
 用户  系统  流逝 
 0.56  0.06 22.17 
#writing submission file
sub_rf <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase =  predict.rforest$predict)
write.csv(sub_rf, file = "sub_rf.csv", row.names = F)
```
做出预测需要22秒。现在是上传提交文件并检查结果的时候了。

![image](https://www.analyticsvidhya.com/wp-content/uploads/2016/05/27-1024x290.png)

这在排行榜上略有改善，但没有预期的那么好。可以再试试GBM，一种增强算法也许可以有所帮助。
### H2O中的GBM
如果您是GBM的新手，我建议您检查本节开始时给出的参考资料。我们可以在H2O中使用简单的代码行来实现GBM:
```r
#GBM
system.time(
  gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 1000, max_depth = 4, learn_rate = 0.01, seed = 1122)
)
|==================================================================================================| 100%
  用户   系统   流逝 
  3.33   0.15 298.91 
```
使用相同数量的树，GBM花费的时间比随机的森林多。花了299秒。您可以使用这个模型来检查这个模型的性能:
```r
h2o.performance (gbm.model)
H2ORegressionMetrics: gbm
** Reported on training data. **
MSE:  6321280
RMSE:  2514.216
MAE:  1859.895
RMSLE:  NaN
Mean Residual Deviance :  6321280
```
正如您所看到的，与前两个模型相比，我们的R方已经有了很大的改进。这显示了一个强大的模式的迹象。让我们做个预测，看看这个模型是否能给我们带来一些改进。
```r
#making prediction and writing submission file
predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
sub_gbm <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.gbm$predict)
write.csv(sub_gbm, file = "sub_gbm.csv", row.names = F)
```
我们已经创建了提交文件。让我们上传它，看看我们是否有任何改进。

![image](https://www.analyticsvidhya.com/wp-content/uploads/2016/05/29-1024x286.png)

我从来没有怀疑过GBM。如果做得好，增加算法通常会得到很好的回报。现在，我将会很有趣地看到我的排行榜:

![image](https://www.analyticsvidhya.com/wp-content/uploads/2016/05/30-1024x111.png)

这是一个巨大的排行榜的飞跃！它就像一个自由落体但安全着陆从第122位到第25位。我们能做得更好吗?也许,我们可以。现在让我们在H2O中使用深度学习算法来提高这个分数。

### H2O中的深度学习
让我简要地概述一下深度学习。在深度学习算法中，存在3层，即输入层、隐藏层和输出层。它的工作原理如下:
1. 我们将数据输入到输入层。
2. 然后，它将数据传输到隐藏层。这些隐藏的层由神经元组成。这些神经元使用一些功能，并协助在变量之间绘制非线性关系。隐藏的层是用户指定的。
3. 最后，这些隐藏层将输出到输出层，然后输出结果。

现在我们来实现这个算法。
```r
#deep learning models
system.time(
  dlearning.model <- h2o.deeplearning(y = y.dep,
                                      x = x.indep,
                                      training_frame = train.h2o,
                                      epoch = 60,
                                      hidden = c(100,100),
                                      activation = "Rectifier",
                                      seed = 1122
  )
)
|==================================================================================================| 100%
  用户   系统   流逝 
  1.06   0.07 167.21 
```
它的执行速度比GBM模型要快。GBM用了约 167秒。隐藏的参数指示这些算法创建每一个隐藏的100个神经元的隐藏层。epoch 负责训练数据的传递的数量。Activation 指的是在整个网络中使用的激活函数。不管怎样，让我们检查一下它的性能。
```r
> h2o.performance(dlearning.model)
H2ORegressionMetrics: deeplearning
** Reported on training data. **
** Metrics reported on temporary training frame with 9881 samples **
MSE:  6171467
RMSE:  2484.244
MAE:  1832.876
RMSLE:  NaN
Mean Residual Deviance :  6171467
```
与GBM模型相比，我们可以看到R方度量的进一步改进。这表明，深度学习模型已经成功地捕获了模型中大量无法解释的差异。我们来做一下预测，看看最终的分数。
```r
#making predictions
predict.dl2 <- as.data.frame(h2o.predict(dlearning.model, test.h2o))
#create a data frame and writing submission file
sub_dlearning <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.dl2$predict)
write.csv(sub_dlearning, file = "sub_dlearning_new.csv", row.names = F)
```
让我们上传最后的提交，并检查分数。

![image](https://www.analyticsvidhya.com/wp-content/uploads/2016/05/32-1024x283.png)

![image](https://www.analyticsvidhya.com/wp-content/uploads/2016/05/33-1024x105.png)

虽然我的分数有所提高，但排名没有。最后，我们通过使用一些特性工程和机器学习算法，最终达到了第25位。我希望你从第154位到第25位的旅程很愉快。如果你一直跟着我到这里，我想你已经准备好再走一步了。

### 下一步怎样改进模型
实际上，你可以做很多事情。在这里，我把它们列出来:
1. 在GBM、深度学习和随机森林中进行参数调优。
2. 使用grid搜索进行参数调优。H2O有一个很好的H2O。完成这个任务的grid
3. 考虑创建更多的特性，这些特性可以为模型带来新的信息。
4. 最后，综合所有的结果以获得一个更好的模型。

## 结束语
我希望你能享受这次使用data.table和H2O的数据探索经历。一旦你熟练使用这两个包，你就可以避免由于内存问题而产生的许多障碍。在本文中，我讨论了使用data.table和H2O实现模型构建的步骤(使用R代码)。虽然，H2O本身可以进行数据处理任务，但我相信data.table是一个非常容易使用的(语法)选项。

在本文中，我的目的是让您开始使用data.table和H2O构建模型。我相信，在这种建模实践之后，您将会变得很有兴趣进一步了解这些包。

这篇文章让你学到了新的东西吗?请在评论中写下你的建议、经验或任何反馈，这些都能让我以更好的方式帮助你。

译自：https://www.analyticsvidhya.com/blog/2016/05/h2o-data-table-build-models-large-data-sets/