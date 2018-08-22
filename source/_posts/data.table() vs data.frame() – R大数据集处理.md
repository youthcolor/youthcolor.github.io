---
title: data.table() vs data.frame() – R大数据集处理
categories: R
tags: [大数据]
---
## 引文
R用户(主要是初学者)在处理大型数据集时往往是比较无助的。他们会被重复的警告以及内存使用不足的错误信息所困扰。他们中的大多数人就会觉得，自己的机器配置不够强大。是时候升级内存，或者在新机器上工作了。你有这样想过吗?
<!--more-->
如果你认真研究过数据集，我相信你一定会有。甚至，当我参加黑色星期五的时候，我也做了同样的工作。数据集包含40多万行。我完全无可奈何。老实说，看到RStudio花费数小时执行一行代码是令人沮丧的。就像我们说的，“需求是发明之母”。我需要一个解决方案。

经过2个小时的互联网搜索研究，我发现了一些有趣的R包*data.table*和api，特别是在不影响执行速度的情况下使用大型数据集。

在本文中，我分享了一种智能方法，当您在大型数据集上工作时，您应该使用这种方法。当您向下滚动时，您将会遇到可以改进您的R编码的更改类型。是时候编写快速而简短的代码了。可以把*data.table*看作是一个关于数据的快速教程。

## 为什么你的机器无法处理大数据集
理解影响您的R代码性能的因素是很重要的。很多时候，你的机器配置强弱直接影响你R代码所能处理的工作。下面是一些妨碍R在大数据集上的性能的例子:

1. 使用*read.csv*加载大文件的csv函数。
2. 使用谷歌chrome:在chrome中打开几个标签会消耗大量系统的内存。这可以在chrome浏览器中使用Shift+Esc键进行检查。(同样也适用于Mozilla web浏览器)
3. 机器配置:R将整个数据集立即读取到RAM中。也就是说，R对象完全存在于内存中。如果您仍在使用2GB RAM机器，那么您在技术上是行不通的。有了2GB内存，就没有足够的空闲RAM空间可以无缝地处理大数据。因此，强烈建议使用至少4GB的机器。
4. 在高温环境下工作:一旦机器升温，处理器速度就会减慢。特别是在夏季，它就会是一个比较严重的问题。

注:我的系统规格是Intel(R) i5-3320M CPU @2.60 GHz，双核，4个逻辑处理器，8GB RAM。
## 关于data.table
*data.table*包是*Matt Dowle*在2008年编写的。

*data.table*作为一个高级版本的data.frame。它从data.frame继承而来，data.frame语法也都可以应用在*data.table*中。这个包可以与任何其他包接受data.frame的包一起使用。

*data.table*的语法与SQL非常相似。因此，如果您使用SQL，您将很快理解它。*data.table*语法的一般形式是:*DT[i, j, by]*，这里

1. DT表示数据表。
2. i <=> where: 表示行索引，这里，是行条件。
3. j <=> select: 表示列索引，这里，是列上放置条件(过滤，总结)。
4. by <=> group_by: 表示任何分类变量，这里，是分组执行的变量。

例如：
```r
#creating a dummy data table
DT <- data.table( ID = 1:50,
                Capacity = sample(100:1000, size = 50, replace = F),
                Code = sample(LETTERS[1:4], 50, replace = T),
                State = rep(c("Alabama","Indiana","Texas","Nevada"), 50))
#simple data.table command
DT[Code == "C", mean(Capacity), State]
     State       V1
1: Alabama 538.8333
2:  Nevada 487.0000
3: Indiana 487.0000
4:   Texas 538.8333
```
让我们看看这个命令是如何工作的。在创建了数据表之后，请求数据表对其Code列为C的行进行过滤，然后要求它计算每一个状态行的Capacity平均值。您不必总是提到语法的所有三个部分。试着在你的结尾执行以下命令:
```r
DT[Code == "D"]
DT[, mean(Capacity), by = State]
DT[Code == "A", mean(Capacity)]
```
## 为什么要使用data.table而不是data.frame
在我深入研究data.table之后，发现了优于data.frame包的几个方面。因此，我建议每个R初学者都可能多的使用data.table。它有很多值得探索的地方。你越早开始使用，你就会做得越好。您应该使用data.table,因为:

1. 在加载数据时，它的速度极快。使用data.table中的fread函数，加载大型数据集只需几秒钟。例如，我使用包含439541行的数据集来检查加载时间。让我们看看fread有多快。
```r
system.time(dt <- read.csv("data.csv"))
user  system elapsed 
11.46  0.21   11.69
system.time(dt <- fread("data.csv"))
user system elapsed 
0.66  0.00   0.66
dim(dt)
[1] 439541 18
```
如您所见，以fread加载数据的速度比基本函数read.csv快16倍。fread()比read.csv()更快，因为，read.csv()尝试首先将行读取到内存中，然后尝试将它们转换为整数，并作为数据类型进行转换。而fread()只是简单地把所有的数据都读成字符。

2. 它甚至比流行的dplyr，plyr软件包还要快。data.table 为任务提供了足够的空间，如聚合、过滤、合并、分组和其他相关任务等。例如:
```r
system.time(dt %>% group_by(Store_ID) %>% filter(Gender == "F") %>%                                       summarise(sum(Transaction_Amount), mean(Var2))) #with dplyr
user system elapsed 
0.13  0.02   0.21
system.time(dt[Gender == "F", .(sum(Transaction_Amount), mean(Var2)), by = Store_ID])
user system elapsed 
0.02  0.00   0.01
```
data.table处理这个任务的速度比dplyr快20倍。之所以发生这种情况，是因为它避免将内存分配给诸如过滤之类的中间步骤。另外，dplyr还创建了作为数据的整个数据帧的深度拷贝。data.table对数据框架进行了简单的复制。浅拷贝意味着数据不会在系统的内存中被物理地复制。它只是一个列指针(名称)的副本。深度拷贝将整个数据复制到内存中的另一个位置。因此，随着内存效率的提高，计算的速度也得到了提高。

3. 不只是读取文件，将数据写入文件方面，data.table也比write.csv()快得多。这个包提供了fwrite()函数，具有并行快速写入能力。所以，下次你要写入100万行数据，试试这个函数。
4. 在构建特征中，如自动索引、滚动连接、重叠范围连接，都进一步增强了用户在大数据集上的体验。

因此，您可以看到data.frame没有什么问题，它只是缺少像data.table一样更适用的特性和操作。
## 重要的数据操作命令
本教程的目的是为您提供一些方便的命令，这些命令可以加速您的建模过程。实际上，在这个包中有很多要探索的地方，您可能会对从哪里开始、何时使用哪个命令以及何时使用特定命令感到困惑。在这里，我提供了一些最常见的问题，这些问题可能是您在进行数据探测/操作时经常遇到的。

下面使用的数据集可以从这里下载:[下载](http://download.geonames.org/export/zip/GB_full.csv.zip)。数据集包含1720673行12列。有趣的是，data.table会使这些数据多长时间被加载，是采取行动的时候了!

注意:数据集包含不均匀分布的观测值，即空列和NA值。获取这些数据的原因是为了检查data.table在处理大数据集的性能。
```r
#set working directory
setwd("C:/Users/Administrator/Desktop/新建文件夹/")
#load data
DT <- fread("GB_full.csv")
Read 1720673 rows and 12 (of 12) columns from 0.191 GB file in 00:00:07
```
读取数据只用了7秒。
### 如何获取行和列的子集
```r
#subsetting rows
sub_rows <- DT[V4 == "England" & V3 == "Beswick"]
   V1   V2      V3      V4  V5                       V6       V7 V8 V9 V10 V11 V12
1: GB YO25 Beswick England ENG East Riding of Yorkshire 11609011        NA  NA   4
#subsetting columns
sub_columns <- DT[,.(V2,V3,V4)]
               V2                  V3       V4
      1:     AB10            Aberdeen Scotland
      2: AB10 1AB            Aberdeen Scotland
      3: AB10 1AF            Aberdeen Scotland
      4: AB10 1AG            Aberdeen Scotland
      5: AB10 1AH            Aberdeen Scotland
     ---                                      
1720669:  ZE3 9JU Shetland South Ward Scotland
1720670:  ZE3 9JW Shetland South Ward Scotland
1720671:  ZE3 9JX Shetland South Ward Scotland
1720672:  ZE3 9JY Shetland South Ward Scotland
1720673:  ZE3 9JZ Shetland South Ward Scotland
```
在data.table中，列被称为变量。因此，我们不需要将变量称为DT$列名，仅列名称就行了。如果你输入DT[,.c(V2，V3，V4)]，它会返回一个列值向量。使用.()符号，将变量封装在list()中，并返回数据表。实际上，每个数据表或数据框都是对长度和不同数据类型的列表的汇编。

设置键可以更快速地获取子集，键不过是更加有效的行名称。如下所示。

### 如何按升序或降序排列变量
```r
#ordering columns
dt_order <- DT[order(V4, -V8)]
```
Order函数是数据表比基本函数order()快得多。原因是，在数据表中的顺序使用基数排序来提高速度。-以降序显示结果。
### 如何在数据集中添加/更新/删除列或值
```r
#add a new column
DT[, V_New := V10 + V11]
```
我们没有将结果返回给DT，这是因为:=操作符通过引用修改输入对象。它会在R中产生较浅的拷贝，从而导致更好的性能，内存占用更少。其结果是不可见的返回。
```r
#update row values
DT[V8 == "Aberdeen City", V8 := "Abr City"]
#delete a column
DT [,c("V6","V7") := NULL ]
```
检查视图(DT)。我们可以看到，数据中包含了数据集中的空白列，可以使用上面的代码删除这些列。实际上，所有这三个步骤都可以在一个命令中完成。这就是命令的链接。
```r
DT[V8 == "Aberdeen City", V8 := "Abr City"][, V_New := V10 + V11][,c("V6","V7") := NULL]
```
### 如何基于对一个列分组来计算变量的函数
让我们计算一下V10变量在V4(显示国家)的基础上的平均值。
```r
#compute the average
DT[, .(average = mean(V1o)), by = V4]
#compute the count
DT[, .N, by = V4]
```
.N是数据中的一个特殊变量。表用于计算变量的计数。如果您希望获得由选项指定的变量的顺序，您可以使用keyby替换by。keyby自动按升序顺序对分组变量进行排序。
### 如何为子集数据设置键值
数据表中的键提供了难以置信的快速结果。我们通常在列名称上设置键，这些键可以是任何类型的，例如数字、因数、整数、字符。一旦一个键被设置为一个变量，它就会以递增的顺序重新排列列的观察值。设置一个键是有帮助的，特别是当您知道需要在一个变量上进行多个计算时。
```r
#setting a key
setkey(DT, V4) 
#subsetting England from V4
DT[.("England")]
```
一旦设置好了键，我们就不再需要一次又一次地提供列名称了。如果我们要在一个列中寻找多个值，我们可以把它写成:
```r
DT[.(c("England", "Scotland"))]
```
类似地，我们也可以设置多个键。这可以用:
```r
setkey(DT, V3, V4)
```
我们可以再次使用这两列的子集值:
```r
DT[.("Shetland South Ward","Scotland")]
```
还可以在上面演示的5个步骤中进行其他一些修改。上面介绍的这5个步骤将帮助您使用data.table执行基本的数据操作任务。为了了解更多，我建议您在每天的R工作中开始使用这个包。你会遇到各种各样的障碍，这就是你的学习曲线会加速的地方。
## 写在最后
本文旨在为您提供一个可以轻松处理大型数据集的路径。你不再需要花钱升级你的机器了，现在是时候升级你处理这种情况的知识了。data.table还有其他几个用于并行计算的包。但是，一旦您精通了data.table，就不需要任何其他的数据操作包了。

在本文中，我讨论了在处理大型数据集时，R中每个初学者都必须知道的一些重要知识。在数据处理之后，接下来的障碍就是模型构建。有了大型数据集，像caret、随机森林、xgboost这样的包需要大量的时间来计算。

我计划下周在我的文章中提供一个有趣的解决方案！在处理大数据时，请告诉我您的痛处。你喜欢看这篇文章吗?在处理大数据集时，您使用了哪些其他的包?在评论中提出您的建议/意见。