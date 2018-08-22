---
title: 使用rvest抓取NBA历史比赛数据
categories: R
tags: [网络爬虫]
---

我们生活在互联网大数据时代，网上遍布各种数据，做数据分析必备的一步就是要能获取想要得到的数据，如果没有数据，做数据分析就如同“巧妇难为无米之炊”。网络爬虫技术为获取数据提供了途径，R提供的*rvest*包，为我们抓取网络数据提供了便利。

下面，以爬取NBA历史比赛数据为例，使用*rvest*进行简单的爬虫实践。
<!--more-->
NBA中文网站 http://www.stat-nba.com/gameList_simple.html 提供了十分全面的NBA历史比赛数据。

![image](http://wx1.sinaimg.cn/mw690/e621a9abgy1flhvqens87j20qw09576k.jpg)

进入比赛数据详情页面，可以看到各种比赛数据记录非常详细。

![image](http://wx4.sinaimg.cn/mw690/e621a9abgy1flhvqkt5n1j20r90e5wgh.jpg)

爬虫的关键，是匹配数据所属html标签。那么，怎样才能比较容易获取到每个数据的html标签呢？虽然利用Chrome浏览器按F12键也分析出元素标签，但是，本人比价喜欢使用Pale Moon浏览器（火狐进化版本），因为用它获取元素的独一无二的识别标签十分方便，无疑就提高了工作效率。

![image](http://wx4.sinaimg.cn/mw690/e621a9abgy1flhw3xj2clj20rd09mgnn.jpg)

选中要分析的元素后，右键下面的该元素标签，即可复制它的独特识别标签，如“div.text:nth-child(1) > div:nth-child(1)”。

![image](http://wx3.sinaimg.cn/mw690/e621a9abgy1flhw42fo7yj205b08mdft.jpg)

下面就有利于我们进一步的爬虫工作了，R代码如下。
```r
# install.packages('rvest')
library('rvest')

# 函数封装，返回dataframe
getNBADataFrame <- function(pageBegin,pageEnd) {
  result <- data.frame()
  for (index in pageBegin:pageEnd) {
    # url
    url <- paste("http://www.stat-nba.com/game/",index,".html",sep="",collapse="")
    webpage <- read_html(url,'uft-8')
    bodytext <- webpage %>% html_node('body') %>% html_text
    # 判断该页面数据是否存在
    if(!length(grep("from coach where id", bodytext, value = FALSE))){
        # 出场 
      chuchang <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(2)') %>%
        html_text %>% sub(pattern = "浜?",replacement = "") %>% as.numeric
        # 投篮
      toulan <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(3)') %>%
        html_text %>% sub(pattern = "%",replacement = "") %>% as.numeric
        # 投篮命中
      mingzhong_t <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(4)') %>%
        html_text %>% as.numeric
        # 投篮出手
      chushou_t <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(5)') %>%
        html_text %>% as.numeric
        # 三分
      sanfen <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(6)') %>%
        html_text %>% sub(pattern = "%",replacement = "") %>% as.numeric
        # 三分命中
      mingzhong_s <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(7)') %>%
        html_text %>% as.numeric
        # 三分出手
      chushou_s <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(8)') %>%
        html_text %>% as.numeric
        # 罚球
      faqiu <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(9)') %>%
        html_text %>% sub(pattern = "%",replacement = "") %>% as.numeric
        # 罚球命中
      mingzhong_f <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(10)') %>%
        html_text %>% as.numeric
        # 罚球出手
      chushou_f <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(11)') %>%
        html_text %>% as.numeric
        # 真实命中
      mingzhong_z <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(12)') %>%
        html_text %>% sub(pattern = "%",replacement = "") %>% as.numeric
        # 篮板
      lanban <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(13)') %>%
        html_text %>% as.numeric
        # 前场
      qianchang <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(14)') %>%
        html_text %>% as.numeric
        # 后场
      houchang <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(15)') %>%
        html_text %>% as.numeric
        # 助攻
      zhugong <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(16)') %>%
        html_text %>% as.numeric
        # 抢断
      qiangduan <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(17)') %>%
        html_text %>% as.numeric
        # 盖帽
      gaimao <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(18)') %>%
        html_text %>% as.numeric
        # 失误
      shiwu <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(19)') %>%
        html_text %>% as.numeric
        # 犯规
      fangui <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(20)') %>%
        html_text %>% as.numeric
        # 得分
      defen <- webpage %>% 
        html_nodes('tr.team_all_content > td:nth-child(21)') %>%
        html_text %>% as.numeric
      date <- webpage %>%
        html_node('#background > div:nth-child(7) > div:nth-child(2)') %>%
        html_text %>% as.Date
        # 转换日期为北京时间
      date <- date + 1
      title <- webpage %>%
        html_node('#background > div:nth-child(7) > div:nth-child(1)') %>%
        html_text %>% strsplit("\n")
        # 赛季
      year <- title[[1]][2]
      # 季前赛or常规赛or季后赛
      type <- title[[1]][3]
      # 客队
      team_a <- webpage %>%
        html_node('div.team:nth-child(1) > div:nth-child(1) > div:nth-child(2) > a:nth-child(1)') %>%
        html_text
        # 主队
      team_h <- webpage %>%
        html_node('div.team:nth-child(3) > div:nth-child(1) > div:nth-child(2) > a:nth-child(1)') %>%
        html_text
        # 客队第一节得分
      defen_1_a <- webpage %>%
        html_node('div.table:nth-child(2) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(2)') %>%
        html_text %>% as.numeric
      defen_2_a <- webpage %>%
        html_node('div.table:nth-child(2) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(2) > td:nth-child(2)') %>%
        html_text %>% as.numeric
      defen_3_a <- webpage %>%
        html_node('div.table:nth-child(2) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(3) > td:nth-child(2)') %>%
        html_text %>% as.numeric
      defen_4_a <- webpage %>%
        html_node('div.table:nth-child(2) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(4) > td:nth-child(2)') %>%
        html_text %>% as.numeric
        # 主队第一节得分
      defen_1_h <- webpage %>%
        html_node('div.table:nth-child(3) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(1)') %>%
        html_text %>% as.numeric
      defen_2_h <- webpage %>%
        html_node('div.table:nth-child(3) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(2) > td:nth-child(1)') %>%
        html_text %>% as.numeric
      defen_3_h <- webpage %>%
        html_node('div.table:nth-child(3) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(3) > td:nth-child(1)') %>%
        html_text %>% as.numeric
      defen_4_h <- webpage %>%
        html_node('div.table:nth-child(3) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(4) > td:nth-child(1)') %>%
        html_text %>% as.numeric
      shengfu <- webpage %>%
        html_node('div.team:nth-child(1) > div:nth-child(1) > div:nth-child(2) > a:nth-child(3)') %>%
        html_text %>% strsplit(" ")
      shengfu <- shengfu[[1]][2] %>% strsplit("胜")
      # 客队胜场
      sheng_a <- shengfu[[1]][1] %>% as.numeric
      shengfu <- shengfu[[1]][2] %>% strsplit("负")
      # 客队负场
      fu_a <- shengfu[[1]][1] %>% as.numeric
      shengfu <- webpage %>%
        html_node('div.team:nth-child(3) > div:nth-child(1) > div:nth-child(2) > a:nth-child(3)') %>%
        html_text %>% strsplit(" ")
      shengfu <- shengfu[[1]][2] %>% strsplit("胜")
      # 主队胜场
      sheng_h <- shengfu[[1]][1] %>% as.numeric
      shengfu <- shengfu[[1]][2] %>% strsplit("负")
      # 主队负场
      fu_h <- shengfu[[1]][1] %>% as.numeric
      # 组合成dataframe格式
      datanew <- data.frame(chuchang_a = chuchang[1],
                            toulan_a = toulan[1],
                            mingzhong_t_a = mingzhong_t[1],
                            chushou_t_a = chushou_t[1],
                            sanfen_a = sanfen[1],
                            mingzhong_s_a = mingzhong_s[1],
                            chushou_s_a = chushou_s[1],
                            faqiu_a = faqiu[1],
                            mingzhong_f_a = mingzhong_f[1],
                            chushou_f_a = chushou_f[1],
                            mingzhong_z_a = mingzhong_z[1],
                            lanban_a = lanban[1],
                            qianchang_a = qianchang[1],
                            houchang_a = houchang[1],
                            zhugong_a = zhugong[1],
                            qiangduan_a = qiangduan[1],
                            gaimao_a = qiangduan[1],
                            shiwu_a = shiwu[1],
                            fangui_a = fangui[1],
                            defen_a = defen[1],
                            
                            chuchang_h = chuchang[2],
                            toulan_h = toulan[2],
                            mingzhong_t_h = mingzhong_t[2],
                            chushou_t_h = chushou_t[2],
                            sanfen_h = sanfen[2],
                            mingzhong_s_h = mingzhong_s[2],
                            chushou_s_h = chushou_s[2],
                            faqiu_h = faqiu[2],
                            mingzhong_f_h = mingzhong_f[2],
                            chushou_f_h = chushou_f[2],
                            mingzhong_z_h = mingzhong_z[2],
                            lanban_h = lanban[2],
                            qianchang_h = qianchang[2],
                            houchang_h = houchang[2],
                            zhugong_h = zhugong[2],
                            qiangduan_h = qiangduan[2],
                            gaimao_h = qiangduan[2],
                            shiwu_h = shiwu[2],
                            fangui_h = fangui[2],
                            defen_h = defen[2],
                            
                            date = date,
                            year = year,
                            type = type,
                            team_a = team_a,
                            team_h = team_h,
                            
                            defen_1_a = defen_1_a,
                            defen_2_a = defen_2_a,
                            defen_3_a = defen_3_a,
                            defen_4_a = defen_4_a,
                            defen_1_h = defen_1_h,
                            defen_2_h = defen_2_h,
                            defen_3_h = defen_3_h,
                            defen_4_h = defen_4_h,
                            
                            sheng_a = sheng_a,
                            fu_a = fu_a,
                            sheng_h = sheng_h,
                            fu_h = fu_h,
                            page = index
      )
      result <- rbind(result, datanew)
      print(index)
      print(dim(result))
    }
  }
  
  return(result)
}
# 保存写入文件（2010年至今所有比赛数据）
data_nba <- getNBADataFrame(27035,40711)
save(data_nba,file="C:/Users/Administrator/Desktop/NBA_Data_Scraping/data_nba.rData")
write.csv(data_nba,file="C:/Users/Administrator/Desktop/NBA_Data_Scraping/data_nba.csv",
          row.names = F,quote = F)
```
运行代码，获取到的数据格式如下

![image](http://wx1.sinaimg.cn/mw690/e621a9abgy1flietkciktj20os04idft.jpg)