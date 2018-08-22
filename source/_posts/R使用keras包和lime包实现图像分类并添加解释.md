---
title: R使用keras包和lime包实现图像分类并添加解释
categories: R
tags: [keras,图像分类,lime,tensorflow]
---
告诉大家一个好消息，我已经使用上keras和TensorFlow了，喜欢它简单直接的建模方式。

## 为不同类型的水果建立图像分类器
本实验的目的是为不同类型的水果建立图像分类器，再次为它建模的快速且简单而惊讶到了，大约100行代码跑了不到一个小时时间！（主代码部分已经做了简单的注释以方便阅读）

这就是我为什么要分享使用keras的原因。

<!--more-->

## 代码开始

如果你之前还没有安装keras，请参照[RStudio’s keras site](https://keras.rstudio.com/)

```r
library(keras)
```
本实验的数据集来自kaggle的[fruit images dataset](https://www.kaggle.com/moltean/fruits/data)。我已经下载并解压了这个数据集，由于我不想对每个水果都建模，因此我只是选取了其中的16个种类的水果进行了建模。

为了尽可能简单，我一开始时定义了几个参数。

```r
# 要建模的水果列表
fruit_list <- c("Kiwi", "Banana", "Apricot", "Avocado", "Cocos", "Clementine", "Mandarine", "Orange",
                "Limes", "Lemon", "Peach", "Plum", "Raspberry", "Strawberry", "Pineapple", "Pomegranate")

# 输出水果类别的数量
output_n <- length(fruit_list)

# 图像大小缩小至20*20像素 (原始图像尺寸为100*100像素)
img_width <- 20
img_height <- 20
target_size <- c(img_width, img_height)

# RGB = 3 通道
channels <- 3

# 图片文件夹路径
train_image_files_path <- "C:/Users/Administrator/Desktop/fruits-360/Training/"
valid_image_files_path <- "C:/Users/Administrator/Desktop/fruits-360/Test/"
```
## 加载图片
```r
# 可选择的数据参数
train_data_gen = image_data_generator(
  rescale = 1/255 #,
  #rotation_range = 40,
  #width_shift_range = 0.2,
  #height_shift_range = 0.2,
  #shear_range = 0.2,
  #zoom_range = 0.2,
  #horizontal_flip = TRUE,
  #fill_mode = "nearest"
)

# 验证数据不应该加入这些参数，除了尺寸上的缩放
valid_data_gen <- image_data_generator(
  rescale = 1/255
  )  
```
## 现在将图片载入内存并调增大小

```r
# 训练集图片
train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                          train_data_gen,
                                          target_size = target_size,
                                          class_mode = "categorical",
                                          classes = fruit_list,
                                          seed = 42)

# 验证集图片
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                          valid_data_gen,
                                          target_size = target_size,
                                          class_mode = "categorical",
                                          classes = fruit_list,
                                          seed = 42)
cat("Number of images per class:")  
## Number of images per class:
table(factor(train_image_array_gen$classes))
##   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15 
## 466 490 492 427 490 490 490 479 490 492 492 447 490 492 490 492
cat("\nClass label vs index mapping:\n")
## Class label vs index mapping:
train_image_array_gen$class_indices
## $Lemon
## [1] 9
## 
## $Peach
## [1] 10
## 
## $Limes
## [1] 8
## 
## $Apricot
## [1] 2
## 
## $Plum
## [1] 11
## 
## $Avocado
## [1] 3
## 
## $Strawberry
## [1] 13
## 
## $Pineapple
## [1] 14
## 
## $Orange
## [1] 7
## 
## $Mandarine
## [1] 6
## 
## $Banana
## [1] 1
## 
## $Clementine
## [1] 5
## 
## $Kiwi
## [1] 0
## 
## $Cocos
## [1] 4
## 
## $Pomegranate
## [1] 15
## 
## $Raspberry
## [1] 12
fruits_classes_indices <- train_image_array_gen$class_indices
save(fruits_classes_indices, file = "C:/Users/Administrator/Desktop/fruits-360/fruits_classes_indices.RData")
```
## 定义模型
接下来，我们定义keras模型。

```r
# 训练集样本的数量
train_samples <- train_image_array_gen$n
# 验证集样本的数量
valid_samples <- valid_image_array_gen$n
# 定义batch大小和epochs数量
batch_size <- 32
epochs <- 10
```
这里我使用的模型是一个简单序列的卷及神经网络，它的隐藏层包含2个卷积层，1个池化层，一个密集层。
```r
# 初始化模型
model <- keras_model_sequential()

# 添加层
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>%
  layer_activation("relu") %>%
  
  # 第二个隐藏层
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%

  # 使用最大池化
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # 扁平最大化过滤输出特征向量并喂给密集层
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%

  # 再喂给输出层
  layer_dense(output_n) %>% 
  layer_activation("softmax")

# 编译
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)
```
## 拟合模型
前面我已经使用了image_data_generator()和flow_images_from_directory()这两个函数，在此，我将使用fit_generator()函数跑训练集。

```r
# 拟合
hist <- model %>% fit_generator(
  # 训练数据
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # 验证数据
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # 打印执行进度
  verbose = 2,
  callbacks = list(
    # 在每一个epoch之后保存最好的模型
    callback_model_checkpoint("C:/Users/Administrator/Desktop/fruits-360/keras/fruits_checkpoints.h5", save_best_only = TRUE),
    # 在TensorBoard中可视化需要
    callback_tensorboard(log_dir = "C:/Users/Administrator/Desktop/fruits-360/keras/logs")
  )
)
```
![image](http://wx3.sinaimg.cn/mw690/e621a9abgy1ftnlxwrqmwj20mr0iswf9.jpg)
在RStudio的视图面板中我们能看到会输出一个交互式图形。 也可以用下面的命令绘制它：

```r
plot(hist)
```
![image](http://wx1.sinaimg.cn/mw690/e621a9abgy1ftnly9jrvyj20mr0isjs7.jpg)
我们可以看到，此模型在验证数据集上相当准确。但是，我们需要清醒认识到我们的图片都很正规，它们都有白色的背景，水果都在图片中央，而且没有其他的东西在图片上。所以，我们的模型对看似不一样的图片将几乎不起作用（这就是为什么我们能在如此小的神经网络实现如此高准确率的原因）。

最后，我想看一下在TensorBoard呈现的TensorFlow图。

```r
tensorboard("C:/Users/Administrator/Desktop/fruits-360/keras/logs")
```
![image](http://wx1.sinaimg.cn/mw690/e621a9abgy1ftnlyn8a8zj211y0hg40o.jpg)
## 疑惑
自己的台式机（Windows 7操作系统）是主频3.4GHz的四核八线程处理器，笔记本（Deepin操作系统）是主频2.6GHz的双核四线程处理器，两者都没有使用GPU加速，结果很费解的是笔记本模型拟合时要比台式机快太多！每个epoch时间：300+ s 对比 10+ s！难道是linux比Windows要快？？？速度对比如下图所示。
- Windows 7

![image](http://wx3.sinaimg.cn/mw690/e621a9abgy1ftnlw4c6l6j20gx0aaaal.jpg)
- Deepin

![image](http://wx1.sinaimg.cn/mw690/e621a9abgy1ftnlwg7fzoj20g90dfjtr.jpg)
# 添加解释
上面做的不仅仅是怎样使用模型预测图片分类，其实单独实现预测是很无聊的，下面我将使用lime包为预测添加解释。
## 加载包和模型

```r
library(keras)   # 调用神经网络
library(lime)    # 解释解释模型
library(magick)  # 预处理图片
library(ggplot2) # 作图
```
加载预训练好的ImageNet模型。

```r
model <- application_vgg16(weights = "imagenet", include_top = TRUE)
model
## Model
______________________________________________________________________________________________________________
Layer (type)                                     Output Shape                                Param #          
==============================================================================================================
input_1 (InputLayer)                             (None, 224, 224, 3)                         0                
______________________________________________________________________________________________________________
block1_conv1 (Conv2D)                            (None, 224, 224, 64)                        1792             
______________________________________________________________________________________________________________
block1_conv2 (Conv2D)                            (None, 224, 224, 64)                        36928            
______________________________________________________________________________________________________________
block1_pool (MaxPooling2D)                       (None, 112, 112, 64)                        0                
______________________________________________________________________________________________________________
block2_conv1 (Conv2D)                            (None, 112, 112, 128)                       73856            
______________________________________________________________________________________________________________
block2_conv2 (Conv2D)                            (None, 112, 112, 128)                       147584           
______________________________________________________________________________________________________________
block2_pool (MaxPooling2D)                       (None, 56, 56, 128)                         0                
______________________________________________________________________________________________________________
block3_conv1 (Conv2D)                            (None, 56, 56, 256)                         295168           
______________________________________________________________________________________________________________
block3_conv2 (Conv2D)                            (None, 56, 56, 256)                         590080           
______________________________________________________________________________________________________________
block3_conv3 (Conv2D)                            (None, 56, 56, 256)                         590080           
______________________________________________________________________________________________________________
block3_pool (MaxPooling2D)                       (None, 28, 28, 256)                         0                
______________________________________________________________________________________________________________
block4_conv1 (Conv2D)                            (None, 28, 28, 512)                         1180160          
______________________________________________________________________________________________________________
block4_conv2 (Conv2D)                            (None, 28, 28, 512)                         2359808          
______________________________________________________________________________________________________________
block4_conv3 (Conv2D)                            (None, 28, 28, 512)                         2359808          
______________________________________________________________________________________________________________
block4_pool (MaxPooling2D)                       (None, 14, 14, 512)                         0                
______________________________________________________________________________________________________________
block5_conv1 (Conv2D)                            (None, 14, 14, 512)                         2359808          
______________________________________________________________________________________________________________
block5_conv2 (Conv2D)                            (None, 14, 14, 512)                         2359808          
______________________________________________________________________________________________________________
block5_conv3 (Conv2D)                            (None, 14, 14, 512)                         2359808          
______________________________________________________________________________________________________________
block5_pool (MaxPooling2D)                       (None, 7, 7, 512)                           0                
______________________________________________________________________________________________________________
flatten (Flatten)                                (None, 25088)                               0                
______________________________________________________________________________________________________________
fc1 (Dense)                                      (None, 4096)                                102764544        
______________________________________________________________________________________________________________
fc2 (Dense)                                      (None, 4096)                                16781312         
______________________________________________________________________________________________________________
predictions (Dense)                              (None, 1000)                                4097000          
==============================================================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
______________________________________________________________________________________________________________
#加载自己的模型
model2 <- load_model_hdf5(filepath = "/home/feng/Downloads/fruits-360/keras/fruits_checkpoints.h5")
model2
## Model
______________________________________________________________________________________________________________
Layer (type)                                     Output Shape                                Param #          
==============================================================================================================
conv2d_1 (Conv2D)                                (None, 20, 20, 32)                          896              
______________________________________________________________________________________________________________
activation_1 (Activation)                        (None, 20, 20, 32)                          0                
______________________________________________________________________________________________________________
conv2d_2 (Conv2D)                                (None, 20, 20, 16)                          4624             
______________________________________________________________________________________________________________
leaky_re_lu_1 (LeakyReLU)                        (None, 20, 20, 16)                          0                
______________________________________________________________________________________________________________
batch_normalization_1 (BatchNormalization)       (None, 20, 20, 16)                          64               
______________________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)                   (None, 10, 10, 16)                          0                
______________________________________________________________________________________________________________
dropout_1 (Dropout)                              (None, 10, 10, 16)                          0                
______________________________________________________________________________________________________________
flatten_1 (Flatten)                              (None, 1600)                                0                
______________________________________________________________________________________________________________
dense_1 (Dense)                                  (None, 100)                                 160100           
______________________________________________________________________________________________________________
activation_2 (Activation)                        (None, 100)                                 0                
______________________________________________________________________________________________________________
dropout_2 (Dropout)                              (None, 100)                                 0                
______________________________________________________________________________________________________________
dense_2 (Dense)                                  (None, 16)                                  1616             
______________________________________________________________________________________________________________
activation_3 (Activation)                        (None, 16)                                  0                
==============================================================================================================
Total params: 167,300
Trainable params: 167,268
Non-trainable params: 32
______________________________________________________________________________________________________________

```
加载准备图片

这里，我将加载并预处理两张水果图片。

- 香蕉

```r
test_image_files_path <- "C:/Users/Administrator/Desktop/fruits-360/Test"
img <- image_read('https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1532608344405&di=1e838eec20a80076760818110dd2e697&imgtype=0&src=http%3A%2F%2Fwww.lp-gj.com%2Fupload%2Fimage%2F20171206%2F20171206094968436843.jpg')
img_path <- file.path(test_image_files_path, "Banana", 'banana.jpg')
image_write(img, img_path)
#plot(as.raster(img))
```
- 橘子

```r
img2 <- image_read('https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1532615118205&di=59abc0a58e9b9a724e3b82b12956a302&imgtype=0&src=http%3A%2F%2Fimgsrc.baidu.com%2Fimgad%2Fpic%2Fitem%2Ffc1f4134970a304ef95645d9dbc8a786c9175c99.jpg')
img_path2 <- file.path(test_image_files_path, "Clementine", 'clementine.jpg')
image_write(img2, img_path2)
#plot(as.raster(img2))
```
Superpixels

> 将图像分割成超像素是生成图像模型解释的重要一步。分割是正确的，并且在图中遵循有意义的模式，而且超级像素的大小/数量是适当的，这两者都很重要。如果图像中的重要特征被切成太多的片段，那么在几乎所有情况下，这些排列可能会破坏图像，从而导致一个糟糕的或失败的解释模型。当感兴趣的对象的大小发生变化时，不可能为超像素的数量设置硬性规则——对象相对于图像的大小越大，生成的超级像素就越少。使用plot_superpixels，在开始使用解释函数之前，可以对超像素参数进行评估。

```r
plot_superpixels(img_path, n_superpixels = 35, weight = 10)
```
![image](http://wx2.sinaimg.cn/mw690/e621a9abgy1ftnl68velpj20dj0a7ta2.jpg)
```r
plot_superpixels(img_path2, n_superpixels = 50, weight = 20)
```
![image](http://wx2.sinaimg.cn/mw690/e621a9abgy1ftnlv8xfe9j20dj0a7jur.jpg)
从超像素的图中我们可以看到，橘子的图像比香蕉图像的分辨率更高。

## 为Imagenet准备图片

```r
image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(224,224))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- imagenet_preprocess_input(x)
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}
```
- 测试预测结果

```r
res <- predict(model, image_prep(c(img_path, img_path2)))
imagenet_decode_predictions(res)
[[1]]
  class_name class_description       score
1  n07753592            banana 0.937849104
2  n03786901            mortar 0.014285510
3  n03532672              hook 0.007741872
4  n04579432           whistle 0.004215840
5  n07747607            orange 0.002990912

[[2]]
  class_name class_description      score
1  n07747607            orange 0.65372097
2  n07749582             lemon 0.07017834
3  n07717556  butternut_squash 0.05920389
4  n07720875       bell_pepper 0.03030883
5  n03937543       pill_bottle 0.02684177
```
- 加载标签和训练解释


```r
model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
explainer <- lime(c(img_path, img_path2), as_classifier(model, model_labels), image_prep)
```
训练这个解释器需要相当长的时间，我自己模型中的较小图片比较大规模的Imagenet训练的时间要快很多。

```r
explanation <- explain(c(img_path, img_path2), explainer, 
                       n_labels = 2, n_features = 35,
                       n_superpixels = 35, weight = 10,
                       background = "white")
```
- plot_image_explanation()函数一次只支持一个实例

```r
plot_image_explanation(explanation)
```
![image](http://wx4.sinaimg.cn/mw690/e621a9abgy1ftnlusl76oj20dj0a7jse.jpg)
```r
clementine <- explanation[explanation$case == "clementine.jpg",]
plot_image_explanation(clementine)
```
![image](http://wx4.sinaimg.cn/mw690/e621a9abgy1ftnlvplzn1j20dj0a7t9y.jpg)
## 为我自己的模型准备图片
测试预测结果(类似于训练和验证图片)

```r
test_datagen <- image_data_generator(rescale = 1/255)

test_generator <- flow_images_from_directory(
        test_image_files_path,
        test_datagen,
        target_size = c(20, 20),
        class_mode = 'categorical')

predictions <- as.data.frame(predict_generator(model2, test_generator, steps = 1))

load("/Users/shiringlander/Documents/Github/DL_AI/Tutti_Frutti/fruits-360/fruits_classes_indices.RData")
fruits_classes_indices_df <- data.frame(indices = unlist(fruits_classes_indices))
fruits_classes_indices_df <- fruits_classes_indices_df[order(fruits_classes_indices_df$indices), , drop = FALSE]
colnames(predictions) <- rownames(fruits_classes_indices_df)

t(round(predictions, digits = 2))
            [,1] [,2]
Kiwi        0.00    0
Banana      0.06    1
Apricot     0.02    0
Avocado     0.00    0
Cocos       0.00    0
Clementine  0.86    0
Mandarine   0.03    0
Orange      0.00    0
Limes       0.00    0
Lemon       0.02    0
Peach       0.00    0
Plum        0.00    0
Raspberry   0.00    0
Strawberry  0.01    0
Pineapple   0.00    0
Pomegranate 0.00    0
for (i in 1:nrow(predictions)) {
  cat(i, ":")
  print(unlist(which.max(predictions[i, ])))
}
1 :Clementine 
         6 
2 :Banana 
     2 
```
尽管这似乎和lime不太符合,所以我准备了类似于Imagenet的图片。

```r
image_prep2 <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(20, 20))
    x <- image_to_array(img)
    x <- reticulate::array_reshape(x, c(1, dim(x)))
    x <- x / 255
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}
```
- 准备标签

```r
fruits_classes_indices_l <- rownames(fruits_classes_indices_df)
names(fruits_classes_indices_l) <- unlist(fruits_classes_indices)
fruits_classes_indices_l
            0             1             2             3             4             5 
       "Kiwi"      "Banana"     "Apricot"     "Avocado"       "Cocos"  "Clementine" 
            6             7             8             9            10            11 
  "Mandarine"      "Orange"       "Limes"       "Lemon"       "Peach"        "Plum" 
           12            13            14            15 
  "Raspberry"  "Strawberry"   "Pineapple" "Pomegranate" 
```
- 训练解释器

```r
explainer2 <- lime(c(img_path, img_path2), as_classifier(model2, fruits_classes_indices_l), image_prep2)
explanation2 <- explain(c(img_path, img_path2), explainer2, 
                        n_labels = 1, n_features = 20,
                        n_superpixels = 35, weight = 10,
                        background = "white")
```
- 绘制特征权重图以寻找绘图的最佳的块阀值 (见下图)

```r
explanation2 %>%
  ggplot(aes(x = feature_weight)) +
    facet_wrap(~ case, scales = "free") +
    geom_density()
```
![image](http://wx2.sinaimg.cn/mw690/e621a9abgy1ftnl5ntfnej20dj0a7mx1.jpg)
- 绘制预测

```r
plot_image_explanation(explanation2, display = 'block', threshold = 1e-09)
```
![image](http://wx2.sinaimg.cn/mw690/e621a9abgy1ftnl5sf0yvj20dj0a7wep.jpg)
```r
clementine2 <- explanation2[explanation2$case == "clementine.jpg",]
plot_image_explanation(clementine2, display = 'block', threshold = 0.01)
```
![image](http://wx4.sinaimg.cn/mw690/e621a9abgy1ftnl5w5tn4j20dj0a7abp.jpg)
```r
sessionInfo()
R version 3.5.0 (2018-04-23)
Platform: x86_64-w64-mingw32/x64 (64-bit)
Running under: Windows 7 x64 (build 7601) Service Pack 1

Matrix products: default

locale:
[1] LC_COLLATE=Chinese (Simplified)_People's Republic of China.936 
[2] LC_CTYPE=Chinese (Simplified)_People's Republic of China.936   
[3] LC_MONETARY=Chinese (Simplified)_People's Republic of China.936
[4] LC_NUMERIC=C                                                   
[5] LC_TIME=Chinese (Simplified)_People's Republic of China.936    

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
[1] ggplot2_2.2.1 magick_1.9    lime_0.4.0    keras_2.1.6  

loaded via a namespace (and not attached):
 [1] Rcpp_0.12.16       pillar_1.2.2       compiler_3.5.0     later_0.7.3       
 [5] gower_0.1.2        plyr_1.8.4         base64enc_0.1-3    iterators_1.0.9   
 [9] tools_3.5.0        zeallot_0.1.0      digest_0.6.15      jsonlite_1.5      
[13] tibble_1.4.2       gtable_0.2.0       lattice_0.20-35    rlang_0.2.0       
[17] Matrix_1.2-14      foreach_1.4.4      shiny_1.1.0        curl_3.2          
[21] parallel_3.5.0     knitr_1.20         htmlwidgets_1.2    grid_3.5.0        
[25] glmnet_2.0-16      reticulate_1.9     R6_2.2.2           magrittr_1.5      
[29] whisker_0.3-2      shinythemes_1.1.1  promises_1.0.1     scales_0.5.0      
[33] codetools_0.2-15   tfruns_1.3         htmltools_0.3.6    abind_1.4-5       
[37] stringdist_0.9.4.7 assertthat_0.2.0   xtable_1.8-2       mime_0.5          
[41] colorspace_1.3-2   httpuv_1.4.5       labeling_0.3       tensorflow_1.8    
[45] stringi_1.1.7      lazyeval_0.2.1     munsell_0.4.3   
```
