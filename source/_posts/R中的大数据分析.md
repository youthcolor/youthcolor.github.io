---
title: R中的大数据分析
categories: R
tags: [大数据,聚类,线性回归]
---

尽管R语言是一门十分强大且健壮的统计型语言，但它最大的短板是对数据大小的限制，因为R需要将数据一次性先加载到内存。R只支持大约4G大小内存的数据量，一旦达到RAM的阀值，操作将无法继续。

但是，R中有一些包很好地支持了大数据分析，比如，*bigmemory*包被广泛用于大数据的统计与计算，还有*biganalytics*,*bigtabulate*,*bigalgebra*等包解决了大数据的管理与统计分析的问题。还一个与*bigmemory*相关联的*ff*包，它支持用户处理大型向量和矩阵，并能同时操作多个大数据文件，*ff*包的有一大好处就是它可以像操作原生的R向量一样进行操作，尽管数据不存储在内存中而是常驻在磁盘中。下面是几个简单的事例。

<!--more-->

## 聚类

### 载入大矩阵

```r
install.packages("bigmemory")
install.packages("biganalytics")
library(bigmemory)
library(biganalytics)
x <- read.big.matrix("FlightTicketData.csv", type = 'integer', header = TRUE,
                  backingfile="data.bin", descriptorfile="data.desc")
head(x)
class(x)
xm <- as.matrix(x)
nrow(xm)
[1] 3156925
```
### 聚类分析


```r
res_bigkmeans <- lapply(1:10, function(i){
  bigkmeans(x, centers = i, iter.max = 50, nstart = 1)
})
class(res_bigkmeans)
lapply(res_bigkmeans, function(x) x$withinss)
var <- sapply(res_bigkmeans, function(x) sum(x$withinss))
var
plot(1:10, var, type = 'b', xlab = "Number of clusters",
     ylab = "Percentage of variance explained")
```
![image](http://wx2.sinaimg.cn/mw690/e621a9abgy1fgmtt5dvc0j20gx0dfmx0.jpg)

```r
res_big <- bigkmeans(x, centers = 3, iter.max = 50, nstart = 1)
res_big

K-means clustering with 3 clusters of sizes 1120691, 919959, 1116275

Cluster means:
         [,1]     [,2]    [,3]     [,4]      [,5]       [,6]     [,7]     [,8]
[1,] 2.757645 11040.08 1104010 30910.66 0.6813850 0.03740460 1.989817 2211.801
[2,] 2.663235 12850.78 1285081 32097.61 0.6323662 0.03459393 2.084982 2305.836
[3,] 2.744241 14513.19 1451322 32768.11 0.6545699 0.02660276 1.974971 2390.292
         [,9]
[1,] 1.949151
[2,] 1.929160
[3,] 1.930394

Clustering vector:
   [1] 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1
  [40] 1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3
  [79] 3 1 1 1 2 3 3 1 1 1 2 2 2 2 2 2 3 3 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 [118] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 [157] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 [196] 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 2 2 2 3 3 3 1 1 1 1 1 2 2 3 3 1 2 2 1 1 1 1
 [235] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3
 [274] 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 3 1 1 1 1 1 1 1 2 2
 [313] 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 [352] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2
 [391] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3
 [430] 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 2
 [469] 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 1 2 3 3 3 3 3 3 3 3 1 1 2 2 1 1 2 2
 [508] 2 2 2 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3
 [547] 3 3 3 3 3 3 2 2 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2
 [586] 2 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 [625] 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1
 [664] 3 1 1 1 1 1 1 1 1 1 1 1 2 2 2 3 3 3 3 3 3 3 3 3 2 3 1 2 2 2 1 2 1 1 1 1 1 1 1
 [703] 1 1 1 1 1 1 1 1 1 1 3 3 1 2 1 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 [742] 3 3 3 3 3 3 3 1 1 2 2 2 2 2 2 3 3 1 1 2 2 2 3 1 1 1 1 1 1 1 1 3 3 1 2 2 3 1 1
 [781] 2 3 3 3 1 3 3 3 3 3 3 3 3 1 3 1 1 1 2 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 [820] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 [859] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 [898] 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 [937] 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 [976] 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1
 [ reached getOption("max.print") -- omitted 3155925 entries ]

Within cluster sum of squares by cluster:
[1] 2.183142e+15 2.010160e+15 2.466224e+15

Available components:

[1] "cluster"  "centers"  "withinss" "size"  
```
## 线性回归分析

###  载入数据


```r
install.packages("ff")
install.packages("biglm")
library(ff)
library(biglm)
download.file("http://www.irs.gov/file_source/pub/irs-soi/12zpallagi.csv", 
              "soi.csv")
x <- read.csv.ffdf(file = "soi.csv", header = TRUE)
class(x)
[1] "ffdf"
head(x)

ffdf (all open) dim=c(73732,6), dimorder=c(1,2) row.names=NULL
ffdf virtual mapping
          PhysicalName VirtualVmode PhysicalVmode  AsIs VirtualIsMatrix
STATEFIPS    STATEFIPS      integer       integer FALSE           FALSE
STATE            STATE      integer       integer FALSE           FALSE
zipcode        zipcode      integer       integer FALSE           FALSE
AGI_STUB      AGI_STUB      integer       integer FALSE           FALSE
N1                  N1       double        double FALSE           FALSE
MARS1            MARS1       double        double FALSE           FALSE
          PhysicalIsMatrix PhysicalElementNo PhysicalFirstCol PhysicalLastCol
STATEFIPS            FALSE                 1                1               1
STATE                FALSE                 2                1               1
zipcode              FALSE                 3                1               1
AGI_STUB             FALSE                 4                1               1
N1                   FALSE                 5                1               1
MARS1                FALSE                 6                1               1
          PhysicalIsOpen
STATEFIPS           TRUE
STATE               TRUE
zipcode             TRUE
AGI_STUB            TRUE
N1                  TRUE
MARS1               TRUE
ffdf data
      STATEFIPS  STATE zipcode AGI_STUB     N1  MARS1
1         1     AL          0     1     889920 490850
2         1     AL          0     2     491150 194370
3         1     AL          0     3     254280  68160
4         1     AL          0     4     160160  23020
5         1     AL          0     5     183320  15880
6         1     AL          0     6      44840   3420
7         1     AL      35004     1       1600    990
8         1     AL      35004     2       1310    570
:             :      :       :        :      :      :
73725    27     MN      55412     1       4820   2890
73726    27     MN      55412     2       3020   1710
73727    27     MN      55412     3       1280    630
73728    27     MN      55412     4        590    200
73729    27     MN      55412     5        440     90
73730    27     MN      55412     6         40      0
73731    27     MN      55413     1       2630   2080
73732    27     MN          5    NA     NA     NA 
```
### 应用线性回归模型

```r
require(biglm)
mymodel <- biglm(A02300 ~ A00200+AGI_STUB+NUMDEP+MARS2, data = x)
summary(mymodel)

Large data regression model: biglm(A02300 ~ A00200 + AGI_STUB + NUMDEP + MARS2, data = x)
Sample size =  73731 
                Coef      (95%     CI)      SE      p
(Intercept) -60.0777 -161.8333 41.6779 50.8778 0.2377
A00200       -0.0014   -0.0015 -0.0014  0.0000 0.0000
AGI_STUB     -2.8467  -28.9734 23.2800 13.0634 0.8275
NUMDEP        1.0457    1.0421  1.0493  0.0018 0.0000
MARS2        -0.3549   -0.3679 -0.3419  0.0065 0.0000

summary(mymodel)$rsq
[1] 0.9424789
```
可覆盖解释了94.24789%的线性回归模型看起来相当不错！