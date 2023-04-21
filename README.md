
显示关系分类器代码
------
### 论文PDTB-Ji

  代码已上传，但是不完整，估计跑不通，代码在文件updown中  
  代码分析放在PDTB-Ji.doc中
  

### 论文DisSent: Learning Sentence Representations from Explicit Discourse Relations
  代码在文件DisExtract-master中，不过好像和显示关系分类器没有直接的关系，更多的是利用这个关系进行隐式分类，如果帮助不太大就忽略这个文件夹吧 
  代码还在阅读  

### 论文PDTB-Lin
  https://github.com/cgpotts/pdtb2  
### Automatic sense prediction for implicit discourse relations in text

情感分类代码
------
### TextBlob库
  效果不理想，它只能针对简单的单词作为情绪的判断，比如love，但是事实上这只是一个活动的主题，并没有体现情绪
  
### Vader库
  基于深度学习的训练模型，https://github.com/cjhutto/vaderSentiment 上提供了一个训练好的模型，可以直接下载使用，由于加入了pos、neg、neu三维的属性，使得在分类上相对细致了一点，通过比较这三者谁的评分更突出判断情感，对于TextBlob库中遇到的上述问题能够解决得很好，但是在更细致而隐形的情感判断，如“令人激动的”这些的决策方面有些保守。但是整体而言还是明显好于TextBlob
  
### Bert
  论文出自N19-1423，代码 https://github.com/gaoliwei1102/csdn_bert_classifier 此代码训练的是中文语料库
  

ChatGPT prompt test
------
  ChatGPT基本能找到情感句所在的位置，基本上语料库中所标注的情感都能找到，但是他会加入一些情感可能没那么强的句子作为情感子句。也有一些错误，比如他会经常把过度/时间标注的句子做欸情感句子输出，我的理解是他可能把几句合成一个情感句，所以都输出了
  
  输入是否带有“,happiness,激动”这样的标签对chatGPT有一定的暗示作用，如果含有这个标签，chatpgt一般能够精确地输出，如果不行，他一般会认为很多句子都是情感句
  
  情感的判断并不如人类感知的那样，比如会出现一些奇怪的形容词，被标记为了情感，但是事实上这不属于我们感知的情感
  
  原因判别：原因的识别chatgpt基本会判定很多句子都属于原因，而一般都很少以单句为原因
  
  原因判别也基本会覆盖人工标注的范围
  
  
  
