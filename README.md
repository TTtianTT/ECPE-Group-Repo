# Leveraging Connectives in ECPE
## Contents
- [Overall Structure](#Overall Structure)  
- [Detailed Work](#Detailed Work)  
  - [Data Mining and Data Preprocessing](#Data Mining and Data Preprocessing)  
  - [Leveraging Connectives](#Leveraging Connectives)  
  - [Explicit Relation Classifier](#Explicit Relation Classifier)  
  - [ECPE-MM-R](#ECPE-MM-R)  
- [Schedule](#Schedule)  
- [Related Works](#Related Works)  
- [Discussion](#Discussion)  
- [Analysis](#Analysis)  
- [Story](#Story)  
- [Contribution](#Contribution)  
- [Future Work](#Future Work)  

## Overall Structure
  ![image](https://github.com/JunfengRan/ECPE-Group-Repo/blob/main/ECPE.jpg)  
  项目主要分为4个部分. 第一部分为挖掘PDTB2.0的(单)连接词 (65 of 100), 统计出现次数和出现的形式(PDTB中的哪种篇章关系, 4-type relations Comparison, Contingency, Expansion, Temporal以及11-type分别统计). 第二部分为跑通ECPE-MM-R原始代码并对其进行改造. 第三部分为显化隐藏连接词并对产生的句子进行打分, 主要过程大致如下: 我们首先判断每一对子句对是否已经存在连接词, 若不存在则添加一个单连接词(在cause子句或emotion子句的开头添加, 感觉最优情况应该是只针对cause子句添加连接词, 需要实验进一步验证). 从产生的若干个新子句对中选取评分最高的子句对作为新的子句对, 然后判断子句对是否存在因果关系, 如果有因果关系则同时将连接词插入回原始的篇章中. 第四部分希望通过在PDTB2.0上标注情感关键词以尝试用情感词增强篇章关系分析的结果, 这一部分将视情况选做. 如果我们后续希望做纯中文的研究, 则会使用HIT-CDTB语料集. 我们也可能会在原有基础上使用跨领域预训练来增强模型, 比如通过收集含有显式连接词的句子作为无监督预训练语料(Giga Words). 后续可能的扩展实验包含使用probe方法研究连接词的语义相关度和类比关系等.

## Detailed Work
### Data Mining and Data Preprocessing

### Leveraging Connectives

### Explicit Relation Classifier

### ECPE-MM-R

## Schedule
  2023.1.4-2023.1.11 尝试跑通ECPE-MM-R原始代码, 使用服务器的同学可以尝试运用Accelerate库或数据并行(https://zhuanlan.zhihu.com/p/467103734).  
  2023.1.11-2023.1.18 挖掘PDTB2.0的(单)连接词 (65 of 100), 统计出现次数和出现的形式(PDTB中的哪种篇章关系), 应该用不了一周.  
  2023.2.8-2023.2.15 添加一个单连接词, 从产生的若干个新子句对中选取评分最高的子句对作为新的子句对, 然后判断子句对是否存在因果关系. 此处需要尝试使用其他显式篇章关系分类器.  
  2023.2.15-2023.2.22 将上述方法与ECPE-MM-R结合得到总体模型并完成实验.  
  2023.2.22-2023.3.1 尝试通过在PDTB2.0上自动标注情感极性以尝试用情感词增强篇章关系分析的结果. 主要做4-type relations Comparison, Contingency, Expansion, Temporal的binary实验(One versus All), top-level one-versus-all binary classification (Pitler et al., 2009).  

## Related Works

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
