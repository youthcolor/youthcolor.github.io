---
title: R机器学习之股票预测初探
categories: R
tags: [股票预测,机器学习,特征选取,量化金融]
---

如今机器学习在工业界和学术界都愈发火热，智能化的崭新时代正在向我们走来。在金融领域，是否也能应用机器学习的方法做股票涨跌的预测呢？下面就以R语言的相关机器学习包为例，简单做股票预测方面的初探。

<!--more-->
## 数据准备

昨天，我已经从同花顺中导出了“*分众传媒*”（股票代码：*002027*）的历史交易数据，该数据集包含了“时间,开盘,最高,最低,收盘,涨幅,振幅,总手,金额,换手%,成交次数”等维度的数据。

```r
# 载入包
library(quantmod);library(TTR);library(randomForest); 
library(caret);library(corrplot);library(pROC);library(FSelector)
XYZQ <- read.csv("002027.csv", header = T, stringsAsFactors = F)
names(XYZQ) <- c("Index", "Open", "High", "Low", "Close", "zhangfu", "zhenfu", 
                 "Volume", "jine", "huanshou", "chengjiaocishu")
XYZQ <- XYZQ[XYZQ$Volume != 0, ]
tail(XYZQ)
             Index   Open   High    Low  Close zhangfu zhenfu      Volume
2979 2017-06-15,四 102.94 104.48 101.04 101.53  -1.77%  3.33%  34,873,418
2980 2017-06-16,五 101.53 103.15 100.27 102.73   1.18%  2.83%  31,332,336
2981 2017-06-19,一  92.55  97.11  92.55  92.76  -9.71%  4.44% 162,266,250
2982 2017-06-20,二  93.11  95.57  93.11  94.23   1.58%  2.65%  44,149,718
2983 2017-06-21,三  94.30  96.41  91.71  94.58   0.37%  4.99%  41,641,757
2984 2017-06-22,四  94.09  97.95  93.88  96.48   2.01%  4.30%  23,065,242
              jine huanshou chengjiaocishu
2979   503,424,470    0.861          23056
2980   448,854,630    0.774          21137
2981 2,141,561,400    4.010          58142
2982   586,119,870    1.090          26975
2983   550,581,210    1.030          24989
2984   313,326,010    0.570          13919
```
## 数据预处理
下面需要对导出的数据进行处理，使他符合自己期望的格式要求。
```r
for(i in 1:nrow(XYZQ)){
  # XYZQ[i,"week"] <-  strsplit(as.character(XYZQ[i,"Index"]), split = ",")[[1]][2]
  # 提取日期
  XYZQ[i,1] <-  strsplit(as.character(XYZQ[i,1]), split = ",")[[1]][1]
  # 去除%号
  XYZQ[i,"zhangfu"] <-  strsplit(as.character(XYZQ[i,"zhangfu"]), split = "%")[[1]][1]
  XYZQ[i,"zhenfu"] <-  strsplit(as.character(XYZQ[i,"zhenfu"]), split = "%")[[1]][1]
}
# 类型转换
XYZQ[,1] <- as.Date(XYZQ[,1])
# XYZQ$week <- as.factor(XYZQ$week)
XYZQ$zhangfu <- as.numeric(XYZQ$zhangfu)
XYZQ$zhenfu <- as.numeric(XYZQ$zhenfu)
XYZQ$Volume <- as.numeric(gsub(",","", XYZQ$Volume))
XYZQ$jine <- as.numeric(gsub(",","", XYZQ$jine))
# 时间序列xts类型转换
XYZQ <- xts(XYZQ[,-1], order.by = XYZQ[, 1])
```
## 特征提取
在R中提供了许多金融指标的*TTR*包，使用它可以很方便的提取出各种金融指标。
```r
# 换手率
huanshou <- XYZQ$huanshou
huanshou <- as.data.frame(huanshou)$huanshou
huanshou <- c(NA, huanshou)
# 振幅
zhenfu <- XYZQ$zhenfu
zhenfu <- as.data.frame(zhenfu)$zhenfu
zhenfu <- c(NA, zhenfu)
# 第二日涨跌信号
Close <- ifelse(XYZQ$zhangfu > 0, 1, 0)
Close <- as.data.frame(Close)$zhangfu
Close <- c(Close,NA)
# TTR指标
myATR <- ATR(HLC(XYZQ))[,'atr'] ; myATR <- c(NA,myATR) ;
mySMI <- SMI(HLC(XYZQ))[,'SMI'] ; mySMI <- c(NA,mySMI) ;
myADX <- ADX(HLC(XYZQ))[,'ADX'] ; myADX <- c(NA,myADX) ;
myAroon <- aroon(XYZQ[,c('High','Low')])$oscillator ; myAroon <- c(NA,myAroon) ;
myBB <- BBands(HLC(XYZQ))[,'pctB'] ; myBB <- c(NA,myBB) ;
myChaikinVol <- Delt(chaikinVolatility(XYZQ[,c("High","Low")]))[,1] ; myChaikinVol <- c(NA,myChaikinVol) ;
myCLV <- EMA(CLV(HLC(XYZQ)))[,1] ; myCLV <- c(NA,myCLV) ;
myEMV <- EMV(XYZQ[,c('High','Low')],XYZQ[,'Volume'])[,2] ; myEMV <- c(NA,myEMV) ;
myMACD <- MACD(Cl(XYZQ))[,2] ; myMACD <- c(NA,myMACD) ;
myMFI <- MFI(XYZQ[,c("High","Low","Close")], XYZQ[,"Volume"]) ; myMFI <- c(NA,myMFI) ;
mySAR <- SAR(XYZQ[,c('High','Close')]) [,1] ; mySAR <- c(NA,mySAR) ;
myVolat <- volatility(OHLC(XYZQ),calc="garman")[,1] ; myVolat <- c(NA,myVolat) ;
myCMO <- CMO(Cl(XYZQ)) ; myCMO <- c(NA,myCMO) ;
myEMA <- EMA(Delt(Cl(XYZQ))) ; myEMA <- c(NA,myEMA) ;
forceindex <- (XYZQ$Close - XYZQ$Open) * XYZQ$Volume ; forceindex <- c(NA,forceindex) ;
WillR5  <- WPR(XYZQ[,c("High","Low","Close")], n = 5) ; WillR5 <- c(NA,WillR5) ;
WillR10 <- WPR(XYZQ[,c("High","Low","Close")], n = 10) ; WillR10 <- c(NA,WillR10) ;
WillR15 <- WPR(XYZQ[,c("High","Low","Close")], n = 15) ; WillR15 <- c(NA,WillR15) ;
WillR30  <- WPR(XYZQ[,c("High","Low","Close")], n = 30) ; WillR30 <- c(NA,WillR30) ;
WillR45 <- WPR(XYZQ[,c("High","Low","Close")], n = 45) ; WillR45 <- c(NA,WillR45) ;
WillR60 <- WPR(XYZQ[,c("High","Low","Close")], n = 60) ; WillR60 <- c(NA,WillR60) ;
RSI5  <- RSI(XYZQ$Close, n = 5,maType="WMA") ;RSI5 <- c(NA,RSI5) ;
RSI10 <- RSI(XYZQ$Close, n = 10,maType="WMA") ;RSI10 <- c(NA,RSI10) ;
RSI15 <- RSI(XYZQ$Close, n = 15,maType="WMA") ;RSI15 <- c(NA,RSI15) ;
RSI30  <- RSI(XYZQ$Close, n = 30,maType="WMA") ;RSI30 <- c(NA,RSI30) ;
RSI45 <- RSI(XYZQ$Close, n = 45,maType="WMA") ;RSI45 <- c(NA,RSI45) ;
RSI60 <- RSI(XYZQ$Close, n = 60,maType="WMA") ;RSI60 <- c(NA,RSI60) ;
ROC5 <- ROC(XYZQ$Close, n = 5,type ="discrete")*100 ; ROC5 <- c(NA,ROC5) ;
ROC10 <- ROC(XYZQ$Close, n = 10,type ="discrete")*100 ; ROC10 <- c(NA,ROC10) ;
ROC15 <- ROC(XYZQ$Close, n = 15,type ="discrete")*100 ; ROC15 <- c(NA,ROC15) ;
ROC30 <- ROC(XYZQ$Close, n = 30,type ="discrete")*100 ; ROC30 <- c(NA,ROC30) ;
ROC45 <- ROC(XYZQ$Close, n = 45,type ="discrete")*100 ; ROC45 <- c(NA,ROC45) ;
ROC60 <- ROC(XYZQ$Close, n = 60,type ="discrete")*100 ; ROC60 <- c(NA,ROC60) ;
MOM5 <- momentum(XYZQ$Close, n = 5, na.pad = TRUE) ; MOM5 <- c(NA,MOM5) ;
MOM10 <- momentum(XYZQ$Close, n = 10, na.pad = TRUE) ; MOM10 <- c(NA,MOM10) ;
MOM15 <- momentum(XYZQ$Close, n = 15, na.pad = TRUE) ; MOM15 <- c(NA,MOM15) ;
MOM30 <- momentum(XYZQ$Close, n = 30, na.pad = TRUE) ; MOM30 <- c(NA,MOM30) ;
MOM45 <- momentum(XYZQ$Close, n = 45, na.pad = TRUE) ; MOM45 <- c(NA,MOM45) ;
MOM60 <- momentum(XYZQ$Close, n = 60, na.pad = TRUE) ; MOM60 <- c(NA,MOM60) ;
ATR5 <- ATR(XYZQ[,c("High","Low","Close")], n = 5, maType="WMA")[,1] ; ATR5 <- c(NA,ATR5) ;
ATR10 <- ATR(XYZQ[,c("High","Low","Close")], n = 10, maType="WMA")[,1]; ATR10 <- c(NA,ATR10) ;
ATR15 <- ATR(XYZQ[,c("High","Low","Close")], n = 15, maType="WMA")[,1]; ATR15 <- c(NA,ATR15) ;
ATR30 <- ATR(XYZQ[,c("High","Low","Close")], n = 30, maType="WMA")[,1] ; ATR30 <- c(NA,ATR30) ;
ATR45 <- ATR(XYZQ[,c("High","Low","Close")], n = 45, maType="WMA")[,1]; ATR45 <- c(NA,ATR45) ;
ATR60 <- ATR(XYZQ[,c("High","Low","Close")], n = 60, maType="WMA")[,1]; ATR60 <- c(NA,ATR60) ;
Mean5 <- runMean(Cl(XYZQ), n = 5);  Mean5 <- c(NA,Mean5) ;
Mean10 <- runMean(Cl(XYZQ), n = 10);  Mean10 <- c(NA,Mean10) ;
Mean15 <- runMean(Cl(XYZQ), n = 15);  Mean15 <- c(NA,Mean15) ;
Mean30 <- runMean(Cl(XYZQ), n = 30);  Mean30 <- c(NA,Mean30) ;
Mean45 <- runMean(Cl(XYZQ), n = 45);  Mean45 <- c(NA,Mean45) ;
Mean60 <- runMean(Cl(XYZQ), n = 60);  Mean60 <- c(NA,Mean60) ;
SD5 <- runSD(Cl(XYZQ), n = 5);  SD5 <- c(NA,SD5) ;
SD10 <- runSD(Cl(XYZQ), n = 10);  SD10 <- c(NA,SD10) ;
SD15 <- runSD(Cl(XYZQ), n = 15);  SD15 <- c(NA,SD15) ;
SD30 <- runSD(Cl(XYZQ), n = 30);  SD30 <- c(NA,SD30) ;
SD45 <- runSD(Cl(XYZQ), n = 45);  SD45 <- c(NA,SD45) ;
SD60 <- runSD(Cl(XYZQ), n = 60);  SD60 <- c(NA,SD60) ;
# 创建特征数据集
mydataset <- data.frame(Close,myATR,mySMI,myADX,myAroon,myBB,myChaikinVol,
                        myCLV,myEMV,myMACD,myMFI,mySAR,myVolat,myCMO,myEMA,
                        forceindex,WillR5,WillR10,WillR15,WillR30,WillR45,WillR60,
                        RSI5,RSI10,RSI15,RSI30,RSI45,RSI60,
                        ROC5,ROC10,ROC15,ROC30,ROC45,ROC60,
                        MOM5,MOM10,MOM15,MOM30,MOM45,MOM60,
                        ATR5,ATR10,ATR15,ATR30,ATR45,ATR60,
                        Mean5,Mean10,Mean15,Mean30,Mean45,Mean60,
                        SD5,SD10,SD15,SD30,SD45,SD60,zhenfu,huanshou)
tail(mydataset)
Close    myATR    mySMI    myADX myAroon      myBB myChaikinVol       myCLV
2980     1 3.541269 59.81687 28.90271      55 0.7459216   -1.1581303 -0.13945017
2981     0 3.494036 58.12533 28.46763      55 0.7118186    0.3881411  0.01469229
2982     1 3.971604 49.35320 27.41221      55 0.3746141   -3.0420371 -0.15305081
2983     1 3.888633 39.81658 26.43217      55 0.3783665    0.2083101 -0.14148355
2984     1 3.946588 31.02765 25.88484      55 0.3630058   -0.8824859 -0.07552716
2985    NA 3.955403 24.34497 24.82386      55 0.4283426   -6.2060752 -0.01131472
             myEMV   myMACD    myMFI     mySAR   myVolat     myCMO         myEMA
2980  6.188251e-04 3.383335 82.26928  98.32502 0.3562983 45.267176  0.0008408088
2981  4.658566e-04 3.476127 77.92001  99.45121 0.3493821 46.711260  0.0028368738
2982  2.671371e-04 3.360701 65.19533 107.71000 0.3824663  9.544950 -0.0153244706
2983 -3.034233e-05 3.136216 63.25320 107.71000 0.3857953  7.727144 -0.0096568672
2984 -4.895251e-04 2.853403 55.06603 107.11200 0.4081373 -4.286628 -0.0072257430
2985 -2.754052e-04 2.573196 52.62331 106.53792 0.3798513 -1.747815 -0.0022594600
     forceindex    WillR5   WillR10   WillR15   WillR30   WillR45   WillR60
2980  -49171519 0.9265367 0.4537445 0.2915094 0.2915094 0.2086428 0.2052474
2981   37598803 0.6693548 0.4055375 0.2580311 0.2349057 0.1681296 0.1653936
2982   34075913 0.9854772 0.9861478 0.7746114 0.7051887 0.5047265 0.4965128
2983   49447684 0.8772827 0.8891821 0.8067026 0.6358491 0.4550979 0.4476918
2984   11659692 0.7752545 0.8206250 0.7857570 0.6193396 0.4432816 0.4360678
2985   55125928 0.5830420 0.7018750 0.7018750 0.5297170 0.4060014 0.3755853
          RSI5    RSI10    RSI15    RSI30    RSI45    RSI60       ROC5     ROC10
2980 21.975737 51.34959 62.96328 63.85533 63.19654 62.64153  -1.168111  5.848624
2981 31.805274 54.59128 64.75479 65.85176 64.42324 63.65843  -3.367510  5.863561
2982  8.578284 23.67412 36.13496 46.71875 50.26581 52.31439 -11.866983 -3.785914
2983 19.109948 27.06896 38.44325 48.65479 51.67119 53.47462 -11.246115 -2.895713
2984 24.012449 25.28028 37.25706 48.82718 51.85549 53.64726  -8.494582 -7.428795
2985 45.294925 33.95349 41.28556 51.58082 53.73532 55.13925  -4.973899 -6.083909
          ROC15     ROC30    ROC45    ROC60   MOM5 MOM10 MOM15 MOM30 MOM45 MOM60
2980 13.5809375  9.125107 19.78528 22.63558  -1.20  5.61 12.14  8.49 16.77 18.74
2981 14.5645143 12.890110 22.32674 23.14793  -3.58  5.69 13.06 11.73 18.75 19.31
2982  2.8837622  4.837251 11.01005 14.06788 -12.49 -3.65  2.60  4.28  9.20 11.44
2983  5.4970891  6.752011 12.67488 18.96225 -11.94 -2.81  4.91  5.96 10.60 15.02
2984  3.3661202  4.014077 13.28303 19.19345  -8.78 -7.59  3.08  3.65 11.09 15.23
2985  0.5838198  4.404285 18.94957 23.66060  -5.05 -6.25  0.56  4.07 15.37 18.46
      ATR5 ATR10 ATR15 ATR30 ATR45 ATR60   Mean5  Mean10   Mean15   Mean30   Mean45
2980  3.44  3.44  3.44  3.44  3.44  3.44 104.524 101.801 98.30533 94.60333 92.10244
2981  2.88  2.88  2.88  2.88  2.88  2.88 103.808 102.370 99.17600 94.99433 92.51911
2982 10.18 10.18 10.18 10.18 10.18 10.18 101.310 102.005 99.34933 95.13700 92.72356
2983  2.81  2.81  2.81  2.81  2.81  2.81  98.922 101.724 99.67667 95.33567 92.95911
2984  4.70  4.70  4.70  4.70  4.70  4.70  97.166 100.965 99.88200 95.45733 93.20556
2985  4.07  4.07  4.07  4.07  4.07  4.07  96.156 100.340 99.91933 95.59300 93.54711
       Mean60      SD5     SD10     SD15     SD30     SD45     SD60 zhenfu huanshou
2980 89.86567 2.046040 3.789094 6.124109 5.798568 6.304757 6.834129   3.33    0.861
2981 90.18750 1.884840 3.402179 5.724033 5.940947 6.374945 6.978591   2.83    0.774
2982 90.37817 5.074185 4.212042 5.465018 5.829461 6.226899 6.887901   4.44    4.010
2983 90.62850 5.024527 4.650943 4.943558 5.687187 6.073691 6.746620   2.65    1.090
2984 90.88233 4.602253 5.161378 4.633946 5.628425 5.903346 6.600000   4.99    1.030
2985 91.19000 3.906678 5.300442 4.601892 5.601426 5.625728 6.418002   4.30    0.570
print(dim(mydataset))
[1] 2985   60
# data.test.tmp <- mydataset[nrow(mydataset),]
mydataset.tmp <- na.omit(mydataset[1:(nrow(mydataset)-1),])
print(dim(mydataset.tmp))
[1] 2923   60
y = mydataset.tmp$Close
cbind(freq=table(y), percentage=prop.table(table(y))*100)
  freq percentage
0 1384   47.34861
1 1539   52.65139
# correlations = cor(mydataset.tmp[,c(2:58)])
# print(head(correlations))
# corrplot(correlations, method="circle")
set.seed(5)
weights <- random.forest.importance(Close~., mydataset.tmp, importance.type = 1)
# 选取前十个权重较大的特征
subset = cutoff.k(weights, 10)
print(subset)
[1] "ROC45"    "huanshou" "WillR60"  "Mean60"   "SD15"     "myCLV"    "zhenfu"  
 [8] "MOM60"    "SD60"     "myATR" 
 
```
## 构建训练集、测试集
用选取出来的十个权重较大的特征构建训练集测试集
```r
dataset_rf = data.frame(Close,mydataset[,subset[1]],mydataset[,subset[2]],
                        mydataset[,subset[3]],mydataset[,subset[4]],
                        mydataset[,subset[5]],mydataset[,subset[6]],
                        mydataset[,subset[7]],mydataset[,subset[8]],
                        mydataset[,subset[9]],mydataset[,subset[10]])

# write.csv(na.omit(dataset_rf),"XYZQ.csv", quote = F, row.names = F)
# 测试集
data.test <- dataset_rf[nrow(dataset_rf),]
# 训练集
dataset_rf <- na.omit(dataset_rf[1:(nrow(dataset_rf)-1),])
dataset_rf$Close <- factor(dataset_rf$Close)
# 十次交叉验证
trainControl = trainControl(method="cv", number=10)
metric = "Accuracy"
```
## 机器学习模型
分别以*nnet*，*naive bayes*，*rpart*，*pcaNNet*，*svmRadial*，*gbm*，*xgbTree*七个模型进行训练测试
```r
# nnet
fit.nnet = train(Close~., data=dataset_rf, method="nnet", 
                 metric=metric, preProc=c("range"),trControl=trainControl)
# naive bayes
fit.nb = train(Close~., data=dataset_rf, method="nb", 
               metric=metric, preProc=c("range"),trControl=trainControl)
# rpart
fit.rpart = train(Close~., data=dataset_rf, method="rpart", 
                  metric=metric,preProc=c("range"),trControl=trainControl)
# pcaNNet
fit.pcaNNet = train(Close~., data=dataset_rf, method="pcaNNet", 
                    metric=metric, preProc=c("range"),trControl=trainControl)
# svmRadial
fit.svmRadial = train(Close~., data=dataset_rf, method="svmRadial", 
                      metric=metric,preProc=c("range"),trControl=trainControl)
# gbm
fit.gbm = train(Close~., data=dataset_rf, method="gbm", 
                metric=metric,preProc=c("range"),trControl=trainControl)
# xgbTree
fit.xgbTree = train(Close~., data=dataset_rf, method="xgbTree", 
                    metric=metric,preProc=c("range"),trControl=trainControl)

## Evaluating the algorithms using the "Accuracy" metric
results = resamples(list(nnet=fit.nnet,nb=fit.nb,rpart=fit.rpart, 
                         pcaNNet=fit.pcaNNet, svmRadial=fit.svmRadial, 
                         gbm=fit.gbm, xgbTree=fit.xgbTree))
summary(results)
Call:
summary.resamples(object = results)

Models: nnet, nb, rpart, pcaNNet, svmRadial, gbm, xgbTree 
Number of resamples: 10 

Accuracy 
               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
nnet      0.5119454 0.5278005 0.5350531 0.5374879 0.5448303 0.5807560    0
nb        0.4914676 0.5179795 0.5435738 0.5371219 0.5593500 0.5616438    0
rpart     0.5034247 0.5239726 0.5299161 0.5289004 0.5376712 0.5426621    0
pcaNNet   0.5034247 0.5226314 0.5273973 0.5299406 0.5439741 0.5582192    0
svmRadial 0.5034247 0.5106918 0.5188356 0.5264996 0.5445205 0.5631399    0
gbm       0.5017065 0.5393836 0.5462329 0.5473935 0.5571672 0.5890411    0
xgbTree   0.5119454 0.5397722 0.5487166 0.5545761 0.5758279 0.5972696    0

Kappa 
                  Min.      1st Qu.     Median       Mean    3rd Qu.       Max. NA's
nnet      -0.018275937  0.019774954 0.03377084 0.03687602 0.05092740 0.12767212    0
nb        -0.054848141 -0.001926439 0.05667858 0.04044339 0.08712856 0.09197804    0
rpart     -0.019553073  0.018301156 0.03064388 0.02607515 0.03487674 0.06389776    0
pcaNNet   -0.037948617  0.001225074 0.02214420 0.02116072 0.04925753 0.07875171    0
svmRadial -0.029068637 -0.010983769 0.00421168 0.01941413 0.05314826 0.09151688    0
gbm       -0.015573809  0.060255860 0.07418475 0.07746468 0.09846185 0.16428163    0
xgbTree    0.004939796  0.062262588 0.07858693 0.09198844 0.13641100 0.17330592    0
```
dotplot(results)
![image](http://wx1.sinaimg.cn/mw690/e621a9abgy1fgva799vyij20hq0cxdhj.jpg)
## 预测涨跌
首先根据每个模型的预测效果的平均数，计算出每个模型的权重，再根据这个权重计算出最终的涨跌概率。有点类似于7个常委进行民主投票进行最终的决策。
```r
pres.pcaNNet <- predict(fit.pcaNNet, data.test)
pres.nnet <- predict(fit.nnet, data.test)
pres.xgbTree <- predict(fit.xgbTree, data.test)
pres.svmRadial <- predict(fit.svmRadial, data.test)
pres.gbm <- predict(fit.gbm, data.test)
pres.nb <- predict(fit.nb, data.test)
pres.rpart <- predict(fit.rpart, data.test)
pres.pcaNNet <- as.numeric(as.character(pres.pcaNNet))
pres.nnet <- as.numeric(as.character(pres.nnet))
pres.xgbTree <- as.numeric(as.character(pres.xgbTree))
pres.svmRadial <- as.numeric(as.character(pres.svmRadial))
pres.gbm <- as.numeric(as.character(pres.gbm))
pres.nb <- as.numeric(as.character(pres.nb))
pres.rpart <- as.numeric(as.character(pres.rpart))
pres <- data.frame(pres.pcaNNet,pres.nnet,pres.xgbTree,pres.svmRadial,pres.gbm,pres.nb,pres.rpart) 
pres
pres.pcaNNet pres.nnet pres.xgbTree pres.svmRadial pres.gbm pres.nb pres.rpart
1            1         1            0              1        0       0          1
# names(getModelInfo())
results.acc <- as.data.frame(summary(results)[3]$statistics$Accuracy)
mean.sum <- sum(results.acc$Mean)
mean.pcaNNet <- results.acc["pcaNNet","Mean"]
mean.nnet <- results.acc["nnet","Mean"]
mean.xgbTree <- results.acc["xgbTree","Mean"]
mean.svmRadial <- results.acc["svmRadial","Mean"]
mean.gbm <- results.acc["gbm","Mean"]
mean.nb <- results.acc["nb","Mean"]
mean.rpart <- results.acc["rpart","Mean"]
# 加入权重计算涨跌概率
q.pcaNNet <- (mean.pcaNNet / mean.sum) * nrow(results.acc)
q.nnet <- (mean.nnet / mean.sum) * nrow(results.acc)
q.xgbTree <- (mean.xgbTree / mean.sum) * nrow(results.acc)
q.svmRadial <- (mean.svmRadial / mean.sum) * nrow(results.acc)
q.gbm <- (mean.gbm / mean.sum) * nrow(results.acc)
q.nb <- (mean.nb / mean.sum) * nrow(results.acc)
q.rpart <- (mean.rpart / mean.sum) * nrow(results.acc)
# 最终的预测结果
p.result <- mean(c(pres.pcaNNet*q.pcaNNet,pres.nnet*q.nnet,pres.xgbTree*q.xgbTree,pres.svmRadial*q.svmRadial,pres.gbm*q.gbm,pres.nb*q.nb,pres.rpart*q.rpart))
p.result
[1] 0.5642939
```
最终计算出的今日涨跌概率为0.56，超出了0.5，那么可以认为，今天的*分众传媒*的预测结果为涨。实际上，分众传媒今天上涨了0.45%，这是巧合吗？有待进一步的探索。如设置止损点、挖掘更有效的特征以及配合蒙特卡洛方法。