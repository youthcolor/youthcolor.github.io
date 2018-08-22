---
title: 实战Caret包对借贷数据进行机器学习预测
categories: R
tags: [机器学习,Caret预测,数据预处理,数据分割,特征选择,参数调优,变量重要性评估]
---
机器学习面临的最大挑战之一该选择什么样的学习算法。在R中，不同的算法会有不同的语法、不同的参数调优和不同的数据格式，因此，对于初学者来说，显得比较吃力。

那么，你如何从一个初学者转变为一个数据科学家，建立数百个模型并把它们堆在一起?当然没有捷径可走，但我想不见得非得把数百个机器学习模型都应用一遍，记住每个算法的不同包名，应用每个算法的语法，为每个算法优化参数等。

Caret(分类和回归训练)包，这可能是R中最大的项目。这个包是我们需要知道的，就像要知道Python机器学习中的Scikit-Learn包一样，它几乎可以解决任何一个受监督的机器学习问题。它为几种机器学习算法提供了统一的接口，并标准化了各种各样的任务，如数据分割、预处理、特性选择、变量重要性评估等。
<!--more-->
今天，我们将探索研究借贷数据预测问题，展示下Caret包的强大功能。PS：尽管Caret确实简化了很多机器学习工作，但是也没必要在它身上投入太多精力。

## 目录
1. 加载数据
2. 数据预处理
3. 数据分割
4. 特征选择
5. 训练模型
6. 参数调优
7. 参数重要性评估
8. 结果预测

## 1. 加载数据
```r
# install.packages("caret", dependencies = c("Depends", "Suggests"))
#Loading caret package
library("caret")
#Loading training data
train<-read.csv("C:\\Users\\Administrator\\Desktop\\loan\\train_u6lujuX_CVtuZ9i.csv",
                stringsAsFactors = T)
#Looking at the structure of caret package.
str(train)
head(rain)
Loan_ID Gender Married Dependents    Education Self_Employed ApplicantIncome CoapplicantIncome
1 LP001002   Male      No          0     Graduate            No            5849                 0
2 LP001003   Male     Yes          1     Graduate            No            4583              1508
3 LP001005   Male     Yes          0     Graduate           Yes            3000                 0
4 LP001006   Male     Yes          0 Not Graduate            No            2583              2358
5 LP001008   Male      No          0     Graduate            No            6000                 0
6 LP001011   Male     Yes          2     Graduate           Yes            5417              4196
  LoanAmount Loan_Amount_Term Credit_History Property_Area Loan_Status
1         NA              360              1         Urban           Y
2        128              360              1         Rural           N
3         66              360              1         Urban           Y
4        120              360              1         Urban           Y
5        141              360              1         Urban           Y
6        267              360              1         Urban           Y
```
## 2. 数据预处理
在建模之前，通常有个数据预处理，先来看看有没有缺失值。
```r
sum(is.na(train))
[1] 86
```
接下来，我们用KNN算法给数据中的缺失值预测并赋值，在这个过程中，对数据进行了标准化操作。
```r
#Imputing missing values using KNN.Also centering and scaling numerical columns
preProcValues <- preProcess(train, method = c("knnImpute","center","scale"))
library('RANN')
train_processed <- predict(preProcValues, train)
sum(is.na(train_processed))
[1] 0
```
使用*dummyVars*函数将分类变量编码为哑变量。
```r
#Converting outcome variable to numeric
train_processed$Loan_Status<-ifelse(train_processed$Loan_Status=='N',0,1)
id<-train_processed$Loan_ID
train_processed$Loan_ID<-NULL
#Converting every categorical variable to numerical using dummy variables
dmy <- dummyVars(" ~ .", data = train_processed,fullRank = T)
train_transformed <- data.frame(predict(dmy, newdata = train_processed))
str(train_transformed)
'data.frame':	614 obs. of  19 variables:
 $ Gender.Female          : num  0 0 0 0 0 0 0 0 0 0 ...
 $ Gender.Male            : num  1 1 1 1 1 1 1 1 1 1 ...
 $ Married.No             : num  1 0 0 0 1 0 0 0 0 0 ...
 $ Married.Yes            : num  0 1 1 1 0 1 1 1 1 1 ...
 $ Dependents.0           : num  1 0 1 1 1 0 1 0 0 0 ...
 $ Dependents.1           : num  0 1 0 0 0 0 0 0 0 1 ...
 $ Dependents.2           : num  0 0 0 0 0 1 0 0 1 0 ...
 $ Dependents.3.          : num  0 0 0 0 0 0 0 1 0 0 ...
 $ Education.Not.Graduate : num  0 0 0 1 0 0 1 0 0 0 ...
 $ Self_Employed.No       : num  1 1 0 1 1 0 1 1 1 1 ...
 $ Self_Employed.Yes      : num  0 0 1 0 0 1 0 0 0 0 ...
 $ ApplicantIncome        : num  0.0729 -0.1343 -0.3934 -0.4617 0.0976 ...
 $ CoapplicantIncome      : num  -0.554 -0.0387 -0.554 0.2518 -0.554 ...
 $ LoanAmount             : num  0.0162 -0.2151 -0.9395 -0.3086 -0.0632 ...
 $ Loan_Amount_Term       : num  0.276 0.276 0.276 0.276 0.276 ...
 $ Credit_History         : num  0.432 0.432 0.432 0.432 0.432 ...
 $ Property_Area.Semiurban: num  0 0 0 0 0 0 0 1 0 1 ...
 $ Property_Area.Urban    : num  1 0 1 1 1 1 1 0 1 0 ...
 $ Loan_Status            : num  1 0 1 1 1 1 1 0 1 0 ...
#Converting the dependent variable back to categorical
train_transformed$Loan_Status<-as.factor(train_transformed$Loan_Status)
```
## 3. 数据分割
为了防止过度拟合，我们以因变量为准将数据分成两份，便于后续的交叉验证。
```r
#Spliting training set into two parts based on outcome: 75% and 25%
index <- createDataPartition(train_transformed$Loan_Status, p=0.75, list=FALSE)
trainSet <- train_transformed[ index,]
testSet <- train_transformed[-index,]
#Checking the structure of trainSet
str(trainSet)
```
## 4. 特征选择
特性选择是建模的极其重要部分，我们将使用递归特性消除方法，以找到用于建模的最佳特性子集。
```r
#Feature selection using rfe in caret
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Loan_Status'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
Loan_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],
                         rfeControl = control)
Loan_Pred_Profile
Recursive feature selection
Outer resampling method: Cross-Validated (10 fold, repeated 3 times) 
Resampling performance over subset size:
 Variables Accuracy  Kappa AccuracySD KappaSD Selected
         4   0.8027 0.4843    0.04251  0.1198         
         8   0.8122 0.5021    0.04127  0.1210         
        16   0.8106 0.5029    0.03919  0.1147         
        18   0.8142 0.5097    0.03693  0.1098        *
The top 5 variables (out of 18):
   Credit_History, Property_Area.Semiurban, CoapplicantIncome, ApplicantIncome, LoanAmount
#Taking only the top 5 predictors
predictors<-c("Credit_History", "LoanAmount", "Loan_Amount_Term", "ApplicantIncome", "CoapplicantIncome")
```
## 5. 训练模型
这可能是Caret从其他任何可用的方案中能脱颖而出的原因，它提供了使用一致的语法实现200多个机器学习算法的能力。如果想查看Caret支持的所有算法的列表，你可以使用*names(getModelInfo())*。
```r
names(getModelInfo())
[1] "ada"                 "AdaBag"              "AdaBoost.M1"         "adaboost"           
[5] "amdai"               "ANFIS"               "avNNet"              "awnb"               
[9] "awtan"               "bag"                 "bagEarth"            "bagEarthGCV"        
[13] "bagFDA"              "bagFDAGCV"           "bam"                 "bartMachine"        
... 
[225] "svmSpectrumString"   "tan"                 "tanSearch"           "treebag"            
[229] "vbmpRadial"          "vglmAdjCat"          "vglmContRatio"       "vglmCumulative"     
[233] "widekernelpls"       "WM"                  "wsrf"                "xgbLinear"          
[237] "xgbTree"             "xyf"
```
我们可以简单地应用大量具有相似语法的算法。例如，应用GBM、随机森林、神经网络、逻辑回归贝叶斯、XGBTree和支持向量机。
```r
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')
model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet')
model_glm<-train(trainSet[,predictors],trainSet[,outcomeName],method='glm')
model_nb<-train(trainSet[,predictors],trainSet[,outcomeName],method='nb')
model_xgbTree<-train(trainSet[,predictors],trainSet[,outcomeName],method='xgbTree')
model_svmRadial<-train(trainSet[,predictors],trainSet[,outcomeName],method='svmRadial')
```
您可以使用参数调优技术进一步优化所有这些算法中的参数。
## 6. 参数调优
使用Caret优化参数是非常容易的。通常，Caret中的参数调优工作如下:

![image](http://wx3.sinaimg.cn/mw690/e621a9abgy1flofbhhvp6j20gv06njss.jpg)

几乎调优过程中的每一个步骤都是可以定制的。在默认情况下使用一组参数来评估模型性能的重新采样技术是引导程序，但是它提供了使用k-交叉、重复k-交叉以及指定可选交叉验证(LOOCV)的替代方法。在本例中，我们将使用5倍的交叉验证重复5次。
```r
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)
```
如果没有定义参数的搜索空间，那么Caret将使用每个可调参数的3个随机值，并使用交叉验证结果来查找该算法的最佳参数集。除此之外，还有两种调优参数的方法：
### 6.1 使用tuneGrid
要找到可以调优的模型的参数，您可以使用
```r
modelLookup(model='gbm')
model         parameter                   label forReg forClass probModel
1   gbm           n.trees   # Boosting Iterations   TRUE     TRUE      TRUE
2   gbm interaction.depth          Max Tree Depth   TRUE     TRUE      TRUE
3   gbm         shrinkage               Shrinkage   TRUE     TRUE      TRUE
4   gbm    n.minobsinnode Min. Terminal Node Size   TRUE     TRUE      TRUE
#Creating grid
grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))
# training the model
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneGrid=grid)
# summarizing the model
print(model_gbm)
Stochastic Gradient Boosting 
461 samples
  5 predictor
  2 classes: '0', '1' 
No pre-processing
Resampling: Cross-Validated (5 fold, repeated 5 times) 
Summary of sample sizes: 368, 369, 370, 368, 369, 369, ... 
Resampling results across tuning parameters:
  shrinkage  interaction.depth  n.minobsinnode  n.trees  Accuracy 
  0.01        1                  3                10     0.6876416
  0.01        1                  3                20     0.6876416
  0.01        1                  3                50     0.8243466
  0.01        1                  3               100     0.8243466
  0.01        1                  3               500     0.8213125
  0.01        1                  3              1000     0.8178246
  0.01        1                  5                10     0.6876416
  0.01        1                  5                20     0.6876416
  ...
 [ reached getOption("max.print") -- omitted 50 rows ]
Accuracy was used to select the optimal model using  the largest value.
The final values used for the model were n.trees = 20, interaction.depth
 = 1, shrinkage = 0.1 and n.minobsinnode = 3.
plot(model_gbm)
```
![image](http://wx1.sinaimg.cn/mw690/e621a9abgy1flofxq76u7j20p00le3yt.jpg)

对于在扩展.grid()中列出的所有参数组合，将使用交叉验证创建和测试一个模型。使用最好的交叉验证性能的参数集将被用来创建最终的模型。
### 6.2. 使用tuneLength
与为每个调优参数指定精确值相反，我们可以简单地要求它通过tuneLength为每个调优参数使用任意数量的可能值。让我们尝试一个使用tuneLength=10的例子。
```r
#using tune length
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=10)
print(model_gbm)
...
Tuning parameter 'shrinkage' was held constant at a value of 0.1
Tuning parameter 'n.minobsinnode' was
 held constant at a value of 10
Accuracy was used to select the optimal model using  the largest value.
The final values used for the model were n.trees = 50, interaction.depth = 1, shrinkage = 0.1
 and n.minobsinnode = 10.
plot(model_gbm)
```
![image](http://wx3.sinaimg.cn/mw690/e621a9abgy1flogag16gwj20p00leweq.jpg)

这里，保持shrinkage和n.minobsinnode参数不变。n.trees和interaction.depth改变10个值，并使用最好的组合来训练最终模型。
## 7. 变量重要性
Caret还通过使用varImp()对任何模型进行变量的评估。让我们来看看我们创建的四个模型的变量的重要性。
```r
#Checking variable importance for GBM
#Variable Importance
varImp(object=model_gbm)
#Plotting Varianle importance for GBM
plot(varImp(object=model_gbm),main="GBM - Variable Importance")
```
![image](http://wx1.sinaimg.cn/mw690/e621a9abgy1flojiw71isj20br0a20sk.jpg)
```r
#Checking variable importance for RF
varImp(object=model_rf)
#Plotting Varianle importance for Random Forest
plot(varImp(object=model_rf),main="RF - Variable Importance")
```
![image](http://wx3.sinaimg.cn/mw690/e621a9abgy1flojlmzgtbj20br0a23yc.jpg)
```r
#Checking variable importance for NNET
varImp(object=model_nnet)
#Plotting Variable importance for Neural Network
plot(varImp(object=model_nnet),main="NNET - Variable Importance")
```
![image](http://wx3.sinaimg.cn/mw690/e621a9abgy1flojnxktmqj20br0a20sk.jpg)
```r
#Checking variable importance for GLM
varImp(object=model_glm)
#Plotting Variable importance for GLM
plot(varImp(object=model_glm),main="GLM - Variable Importance")
```
![image](http://wx3.sinaimg.cn/mw690/e621a9abgy1flojqfcdgyj20br0a2744.jpg)

很明显，不同模型的变量重要性是不同的。从不同的模型中得到的变量重要性的两个主要用途是:
- 对大多数模型来说重要的预测因子代表着真正重要的预测因子。
- 组合预测中，我们应该使用那些具有显著不同重要性的模型进行预测，因为他们的预测也会有所不同。但必须确保所有这些都是足够准确的。
## 8. 结果预测
为了预测测试集的相关变量，Caret提供了predict.train()函数。您需要指定模型名称、测试数据。对于分类问题，Caret还提供了另一个名为*type*的特性，可以设置为*prob*或*raw*。对于类型*raw*，预测只是测试数据的结果类，而对于类型*prob*，它将给出在不同类型的结果变量中出现的每个观察结果的概率。

让我们来看看我们的GBM模型的预测:
```r
#Predictions
predictions<-predict.train(object=model_gbm,testSet[,predictors],type="raw")
confusionMatrix(predictions,testSet[,outcomeName])
Confusion Matrix and Statistics
          Reference
Prediction   0   1
         0  14   2
         1  34 103
               Accuracy : 0.7647          
                 95% CI : (0.6894, 0.8294)
    No Information Rate : 0.6863          
    P-Value [Acc > NIR] : 0.02055         
                  Kappa : 0.3328          
 Mcnemar's Test P-Value : 2.383e-07       
            Sensitivity : 0.2917          
            Specificity : 0.9810          
         Pos Pred Value : 0.8750          
         Neg Pred Value : 0.7518          
             Prevalence : 0.3137          
         Detection Rate : 0.0915          
   Detection Prevalence : 0.1046          
      Balanced Accuracy : 0.6363          
       'Positive' Class : 0 
```
RF的预测：
```r
#Predictions
predictions<-predict.train(object=model_rf,testSet[,predictors],type="raw")
confusionMatrix(predictions,testSet[,outcomeName])
Confusion Matrix and Statistics
          Reference
Prediction  0  1
         0 17  6
         1 31 99
               Accuracy : 0.7582          
                 95% CI : (0.6824, 0.8237)
    No Information Rate : 0.6863          
    P-Value [Acc > NIR] : 0.0315          
                  Kappa : 0.3459          
 Mcnemar's Test P-Value : 7.961e-05       
            Sensitivity : 0.3542          
            Specificity : 0.9429          
         Pos Pred Value : 0.7391          
         Neg Pred Value : 0.7615          
             Prevalence : 0.3137          
         Detection Rate : 0.1111          
   Detection Prevalence : 0.1503          
      Balanced Accuracy : 0.6485          
       'Positive' Class : 0
```
NNET的预测：
```r
#Predictions
predictions<-predict.train(object=model_nnet,testSet[,predictors],type="raw")
confusionMatrix(predictions,testSet[,outcomeName])
Confusion Matrix and Statistics
          Reference
Prediction   0   1
         0  14   2
         1  34 103
               Accuracy : 0.7647          
                 95% CI : (0.6894, 0.8294)
    No Information Rate : 0.6863          
    P-Value [Acc > NIR] : 0.02055         
                  Kappa : 0.3328          
 Mcnemar's Test P-Value : 2.383e-07       
            Sensitivity : 0.2917          
            Specificity : 0.9810          
         Pos Pred Value : 0.8750          
         Neg Pred Value : 0.7518          
             Prevalence : 0.3137          
         Detection Rate : 0.0915          
   Detection Prevalence : 0.1046          
      Balanced Accuracy : 0.6363          
       'Positive' Class : 0
```
GLM的预测：
```r
#Predictions
predictions<-predict.train(object=model_glm,testSet[,predictors],type="raw")
confusionMatrix(predictions,testSet[,outcomeName])
Confusion Matrix and Statistics
          Reference
Prediction   0   1
         0  14   2
         1  34 103
               Accuracy : 0.7647          
                 95% CI : (0.6894, 0.8294)
    No Information Rate : 0.6863          
    P-Value [Acc > NIR] : 0.02055         
                  Kappa : 0.3328          
 Mcnemar's Test P-Value : 2.383e-07       
            Sensitivity : 0.2917          
            Specificity : 0.9810          
         Pos Pred Value : 0.8750          
         Neg Pred Value : 0.7518          
             Prevalence : 0.3137          
         Detection Rate : 0.0915          
   Detection Prevalence : 0.1046          
      Balanced Accuracy : 0.6363          
       'Positive' Class : 0
```
