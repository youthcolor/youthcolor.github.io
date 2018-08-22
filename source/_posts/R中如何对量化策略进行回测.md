---
title: R中如何对量化策略进行回测
categories: R
tags: [量化金融,回测]
---

R中提供了极为便利的回测方法，如*PerformanceAnalytics*包，下面是简短实例。
以A股福田汽车*600166.SS*为例。
<!--more-->

## 第1步: 获取股票数据
```r
setSymbolLookup(MYSTOCK=list(name='600166.SS',src='yahoo',from='2007-01-01'))
getSymbols('MYSTOCK')
```
## 第2步: 建立指标
```r
MYSTOCK <- na.omit(MYSTOCK)
dvi <- DVI(Cl(MYSTOCK))
```
## 第3步: 构建交易规则
```r
sig <- Lag(ifelse(dvi$dvi < 0.5, 1, -1))
```
## 第4步: 交易规则/资金曲线
```r
ret <- ROC(Cl(MYSTOCK))*(sig-0.00025)
ret <- ret['2013-09-01/2014-06-01']
eq <- exp(cumsum(ret))
plot(eq)
```
![image](http://wx2.sinaimg.cn/mw690/e621a9abgy1fi34qkv1qmj20gx09kwem.jpg)
## 第5步: 评估量化策略的表现
```r
table.Drawdowns(ret, top=10)
        From     Trough         To   Depth Length To Trough Recovery
1 2014-01-29 2014-02-18 2014-03-31 -0.1574     39        10       29
2 2013-09-02 2014-01-13 2014-01-28 -0.1246     99        88       11
3 2014-04-23 2014-05-09 2014-05-19 -0.0484     17        11        6
4 2014-04-01 2014-04-03 2014-04-08 -0.0206      5         3        2
5 2014-04-09 2014-04-14 2014-04-15 -0.0163      5         4        1
6 2014-05-22 2014-05-22 2014-05-26 -0.0061      3         1        2
7 2014-05-27 2014-05-27 2014-05-28 -0.0060      2         1        1
8 2014-05-20 2014-05-20 2014-05-21 -0.0041      2         1        1
table.DownsideRisk(ret)
                              600166.SS.Close
Semi Deviation                         0.0147
Gain Deviation                         0.0119
Loss Deviation                         0.0153
Downside Deviation (MAR=210%)          0.0188
Downside Deviation (Rf=0%)             0.0143
Downside Deviation (0%)                0.0143
Maximum Drawdown                       0.1574
Historical VaR (95%)                  -0.0268
Historical ES (95%)                   -0.0466
Modified VaR (95%)                    -0.0334
Modified ES (95%)                     -0.0617
charts.PerformanceSummary(ret)
```
![image](http://wx1.sinaimg.cn/mw690/e621a9abgy1fi34qlhqkdj20m80cjq3d.jpg)