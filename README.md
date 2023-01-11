# Leveraging Connectives in ECPE
## Contents
- [Overall Structure](#Overall-Structure)  
- [Detailed Work](#Detailed-Work)  
  - [Data Mining and Data Preprocessing](#Data-Mining-and-Data-Preprocessing)  
  - [Leveraging Connectives](#Leveraging-Connectives)  
  - [Explicit Relation Classifier](#Explicit-Relation-Classifier)  
  - [ECPE-MM-R](#ECPE-MM-R)  
- [Schedule](#Schedule)  
- [Related Works](#Related-Works)  
  - [Dataset](#Dataset)
  - [Connectives](#Connectives)
  - [ECPE](#ECPE)
- [Discussion](#Discussion)  
- [Analysis](#Analysis)  
- [Story](#Story)  
- [Contribution](#Contribution)  
- [Future Work](#Future-Work)  

## Overall Structure
  ![image](https://github.com/JunfengRan/ECPE-Group-Repo/blob/main/ECPE.jpg)  
  
  The project consists of 4 parts. In the first part, we need to do some data mining and data preprocessing on our corpus. In the second part, we try to figure out how to leverage connectives in ECPE. In the third part, we do our experiments in a classic way using explicit relation classifier. In the last part, we check our improvement on the basis of SOTA.  
  
  We use several datasets. For Chinese studies, we use the HIT-CDTB and ECPE dataset. For English studies, we use the PDTB2.0, PDTB3.0, Gigaword and ECPE-ENG dataset we translate.  
  
## Detailed Work
### Data Mining and Data Preprocessing
  We need to do research on the connectives first. We just simply consider the connectives which is shown in structures like (arg1, conn, arg2) (caution: (conn1, arg1, conn2, arg2) is incorporated into account). While dealing with ECPE dataset, we follow two styles of statistics. One just simply consider the connectives which is shown in structures like (arg1, conn, arg2) like above, the other should consider the following three types (arg1, conn, arg2), (conn, arg1, arg2), (conn1, arg1, conn2, arg2).
  
  ![image](https://github.com/JunfengRan/ECPE-Group-Repo/blob/main/PDTB.png)  
  ![image](https://github.com/JunfengRan/ECPE-Group-Repo/blob/main/PDTB-lv2.png)
  
  For HIT-CDTB dataset and PDTB2.0/3.0 dataset, we want to know their occurence frequency with different discourse relation (top 2 levels) and discourse relation distance for every connectives repectivesly.  The top 11 relations on the second level are Comparison.Concession, Comparison.Contrast, Contingency.Cause, Contingency.Pragmatic cause, Expansion.Alternative, Expansion.Conjunction, Expansion.Instantiation, Expansion.List, Expansion.Restatement, Temporal.Asynchronous, Temporal.Synchrony. Maybe we can set a 2% threshold to count less relation types (need further discussion).
  
  For ECPE dataset, we may need to translate it into English for English studies. For this purpose, we can translate the corpus clause by clause using translate software or algorithms (need further discussion) and then correct the output manually. For dataset itself, we want to know the occurence frequence of connectives in ECP (two styles) and distance (+/-, cause minus emotion) between emotion clause and cause clause for every connectives repectivesly.  
  
  If we want to enhance the performance of the explicit relation classifier, we can use the Gigaword dataset. For this dataset, we need to cleanse the data to seperate clauses and only keep those like (arg1, conn, arg2).  
  
### Leveraging Connectives
  At first, we find the emotion clause using ECPE-MM-R turn 1. And we pair up each emotional clause and each non-emotional clause. Then, we add connectives selected by our data mining work and calculate the probability over the Language Model. Finally, if the sum of probability of causal connectives (probability times percentage) is the greatest, we add connectives with probability into original corpus and process them seperately (maybe choose top 5). However, the loss of this step is unclear (need further discussion). I prefer to say that this step is a automatic step completed by LM so there is no loss we can use.  
  
  On the other hand, we could use the BLEU or ROUGE to score the connectives directly. We will try this in the future.  
  
  For simple verification without whole analysis of PDTB/HIT-CDTB, we use the following list of connectives (need further discussion): Not given yet.
  
### Explicit Relation Classifier
  We can judge the relation of the new pairs of (emotion arg, connectives, arg) with explicit relation classifier as benchmark. The concrete way is not decided yet (need further discussion).  

### ECPE-MM-R
  Last, we can modify the source code of ECPE-MM-R and try to get the SOTA.  

## Schedule
  2023.1.4-2023.1.11 Try to run through the ECPE-MM-R source code. For those using the server can try to use the Accelerate library or data parallelism (https://zhuanlan.zhihu.com/p/467103734).  
  
  2023.1.11-2023.1.18, 2023.2.1-2023.2.8 (not forced)  Analyze the ECPE-MM-R source code and modify it to output the precision and recall rate of every turn. Try to create the framework of adding connectives with probability.  
  
  2023.2.8-2023.2.15 Refine the code above. Complete data mining and data preprocessing.  
  
  2023.2.15-2023.2.22 Fuse the method above with modified ECPE-MM-R source code to complete our experiment.  
  
  2023.2.22-2023.3.1 We can judge the relation of the new pairs of (emotion arg, connectives, arg) with explicit relation classifier as benchmark.  

## Related Works
### Dataset
HIT-CDTB  
http://ir.hit.edu.cn/hit-cdtb/  

PDTB2.0  
http://www.ldc.upenn.edu/Catalog/CatalogEntry.jsp?catalogId=LDC2008T05 (not available)  
https://github.com/cgpotts/pdtb2  

ECPE  
http://hlt.hitsz.edu.cn/?page_id=694 (not available)  
https://github.com/zhoucz97/ECPE-MM-R  

PDTB3.0  
https://catalog.ldc.upenn.edu/LDC2019T05  

Gigaword  
https://catalog.ldc.upenn.edu/LDC2011T07  

### Connectives

### ECPE

## Discussion


## Analysis
  We need to do robust analysis and error analysis. And we need to write something about special examples, like action/trigger words and cascade events.  
  We can also do probe experiments on word meaning relevancy and word analogy of connectives.  

## Story
  In some scenarios, in addition to extracting objects and corresponding emotions, deeper point of view mining also requires extracting the causes that cause emotions to assist in subsequent more complex causal analysis.  
  Emotional cause pair extraction is a task that strongly relies on context, the system needs to model the relationship between emotion and cause, and the cause of emotion in many scenarios may not be in the emotional sentence itself, for example, the reason why multiple rounds of dialogue trigger a certain emotion may be the previous rounds.  
  Emotional cause pair extraction is also a key task for opinion mining interpretability. Emotional reasoning digs into the specific causes of emotions. The reasoned emotional causes will be used as interpretable information to complement some existing tasks, which will greatly promote both the analytical platform and the dialogue system.  

## Contribution
  1.We deliver a simple but powerful method to extract the emotion-cause pair, which is explicating the implicit connectives in the original corpus and then extract the pairs by explicit course relation classifier.  
  2.We develop our model based on a novel and reusable multi-turn QA model.  
  3.We achieve two SOTA by Leveraging Connectives in ECPE and Leveraging Emotion in IDRR.  

## Future Work
  1.For natural language generation task, we can add previously unexistable connectives, reason through the logical chain or template, and then delete these connectives.  
  2.We want to use our method to do more experiments about implicit chapter relation classification, refined connective word selection, and probing connectives representation.  
  3.We'd like to study the evolutionary patterns of emotional dynamics. Add the implicit connective mining task to explicitly let the model learn the dynamic characteristics of emotions, build models based on context, time series, and event information, and compare their effects on the dynamic evolution of emotions.  
  4.We believe the temporal relations are event-driven. We want to do experiments about trigger words in ECPE and make use of TempEval.  
尝试通过在PDTB2.0上自动标注情感极性以尝试用情感词增强篇章关系分析的结果. 主要做4-type relations Comparison, Contingency, Expansion, Temporal的binary实验(One versus All), top-level one-versus-all binary classification (Pitler et al., 2009).
第四部分希望通过在PDTB2.0上标注情感关键词以尝试用情感词增强篇章关系分析的结果, 这一部分将视情况选做. 如果我们后续希望做纯中文的研究, 则会使用HIT-CDTB语料集. 我们也可能会在原有基础上使用跨领域预训练来增强模型, 比如通过收集含有显式连接词的句子作为无监督预训练语料(Giga Words). 后续可能的扩展实验包含使用probe方法研究连接词的语义相关度和类比关系等.
