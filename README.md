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
  
  The project consists of 4 parts. In the first part, we need to do some data mining and data preprocessing on our corpus. In the second part, we try to figure out how to leverage connectives in ECPE. In the third part, we do our experiments in a classic way using explicit relation classifier. In the last part, we check our improvement on the basis of previous SOTA.  
  
  We use several datasets. For Chinese studies, we use the HIT-CDTB and ECPE dataset. For English studies, we use the PDTB2.0, PDTB3.0, Gigaword and ECPE-ENG dataset we translate.  
  
  Since Chinese has only a few connectives, We choose to accomplish English study first.  
  
## Detailed Work

### Data Mining and Data Preprocessing
  We need to do research on the connectives first.  
  
  ![image](https://github.com/JunfengRan/ECPE-Group-Repo/blob/main/PDTB.png)  
  ![image](https://github.com/JunfengRan/ECPE-Group-Repo/blob/main/PDTB-lv2.png)
  
  For HIT-CDTB dataset and PDTB2.0/3.0 dataset, we just simply consider the connectives which are shown in structures like (arg1, conn, arg2) (caution: structures like (conn1, arg1, conn2, arg2) are incorporated into account). And we want to know their occurence frequency with different discourse relation (top 2 levels with reason and result) and discourse relation distance for every connectives repectivesly. The top 11 relations on the second level are Comparison.Concession, Comparison.Contrast, Contingency.Cause, Contingency.Pragmatic cause, Expansion.Alternative, Expansion.Conjunction, Expansion.Instantiation, Expansion.List, Expansion.Restatement, Temporal.Asynchronous, Temporal.Synchrony. We use these top 11 relation with reason and result as our tag.  
  
  |Connective|top level|second level|distance|frequency|
  |:---:|:---:|:---:|:---:|:---:|
  |because|Contingency|Cause.reason|1|1000|
  |because|Contingency|Cause.reason|2|10|
  |while|Temporal|Synchrony|1|100|
  |or|Expansion|Alternative|1|100|
  
  For ECPE dataset, we need to translate it into English for English studies. For this purpose, we can translate the corpus clause by clause using translate software or algorithms and then correct the output manually. While dealing with ECPE dataset, we follow the following style of statistics. We should consider the following three types (arg1, conn, arg2), (conn, arg1, arg2), (conn1, arg1, conn2, arg2). For dataset itself, we want to know the occurence frequence of connectives in ECP (two styles) and distance (+/-, cause minus emotion) between emotion clause and cause clause for every connectives repectivesly.  
  
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
  http://www.ldc.upenn.edu/Catalog/CatalogEntry.jsp?catalogId=LDC2008T05 (origin, not available)  
  https://github.com/cgpotts/pdtb2  
  
  ECPE  
  http://hlt.hitsz.edu.cn/?page_id=694 (origin, not available)  
  https://github.com/zhoucz97/ECPE-MM-R  
  
  PDTB3.0  
  https://catalog.ldc.upenn.edu/LDC2019T05  
  
  Gigaword  
  https://catalog.ldc.upenn.edu/LDC2011T07  
  
### Connectives
  Main reference:  
  Let’s be explicit about that: Distant supervision for implicit discourse relation classification via connective prediction.  
  
  Other reference:  
  Adapting BERT to Implicit Discourse Relation Classification with a Focus on Discourse Connectives.  
  Improving the Inference of Implicit Discourse Relations via Classifying Explicit Discourse Connectives.  
  
  Lin, Z., Kan, M.-Y., and Ng, H. T. (2009). Recognizing implicit discourse relations in the Penn Discourse Treebank. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, pages 343–351.
  Ji, Y. and Eisenstein, J. (2015). One vector is not enough: Entity-augmented distributed semantics for discourse relations. Transactions of the Association of Computational Linguistics, 3:329–344.
### ECPE
  Main reference:  
  A Multi-turn Machine Reading Comprehension Framework with Rethink Mechanism for Emotion-Cause Pair Extraction.  
  
  Other reference:  
  Event-Driven Emotion Cause Extraction with Corpus Construction.  
  Emotion-Cause Pair Extraction: A New Task to Emotion Analysis in Texts.  
  
## Discussion
  1.The original ECPE-MM-R paper didn't give the precision and recall rate of three turns.  
  
  2.I think emotion clause itself cannot be the cause of the emotion.
  we can translate the corpus clause by clause using translate software or algorithms
  3.Why choose delta = 0.7? More theoretical support is needed.  
  
## Analysis
  1.We need to do robust analysis and error analysis. And we need to write something about special examples, like action/trigger words and cascade events.  
  
  2.Creatively, we can do ablation experiments.  
  
  3.We can also do probe experiments on word meaning relevancy and word analogy of connectives.  
  
## Story
  In some scenarios, in addition to extracting objects and corresponding emotions, deeper point of view mining also requires extracting the causes that cause emotions to assist in subsequent more complex causal analysis.  
  
  Emotional cause pair extraction is a task that strongly relies on context, the system needs to model the relationship between emotion and cause, and the cause of emotion in many scenarios may not be in the emotional sentence itself, for example, the reason why multiple rounds of dialogue trigger a certain emotion may be the previous rounds.  
  
  Emotional cause pair extraction is also a key task for opinion mining interpretability. Emotional reasoning digs into the specific causes of emotions. The reasoned emotional causes will be used as interpretable information to complement some existing tasks, which will greatly promote both the analytical platform and the dialogue system.  
  
## Contribution
  1.We deliver a simple but powerful method to extract the emotion-cause pair, which is explicating the implicit connectives in the original corpus and then can extract the pairs by explicit course relation classifier.  
  
  2.Based on the MRC formalization, We develop our model based on a novel and reusable query model.  
  
  3.We demonstrate that our proposed method outperforms existing state-of-the-art performance.  
  
## Future Work
  1.For natural language generation task, we can add previously unexistable connectives, reason through the logical chain or template, and then delete these connectives.  
  
  2.We want to use our method to do more experiments about implicit chapter relation classification, refined connective word selection, and probing connectives representation.  
  
  3.We'd like to study the evolutionary patterns of emotional dynamics. Add the implicit connective mining task to explicitly let the model learn the dynamic characteristics of emotions, build models based on context, time series, and event information, and compare their effects on the dynamic evolution of emotions.  
  
  4.We believe the temporal relations are event-driven. We want to do experiments about trigger words in ECPE and make use of TempEval.  
  
  5.We want to enhance the results of discourse relation analysis on PDTB by automatically labeling the emotional polarity and using our model to identify cause relation in the discourse.
  
  6.We will use cross-domain pre-training to enhance the model on the original basis, such as by collecting sentences with explicit connectives as unsupervised pre-training corpus.  
  
  7.Possible subsequent extended experiments include using the probe method to study the semantic relevance and analogy of connectives.  
  