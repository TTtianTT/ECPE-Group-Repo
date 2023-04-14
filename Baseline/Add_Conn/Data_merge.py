import pandas as pd
import numpy as np
import csv

df_pairs = pd.read_csv('../data/pairs_withconn.csv')
df_discourse = pd.read_csv('../data/discourse.csv')

connLists = []
header = 0

for i in range(len(df_discourse)):
    discourse_connLists = []
    for j in range(len(eval(df_discourse['emotion_pos'][i]))):
        emotion_connList = []
        for k in range(int(df_discourse['doc_len'][i])):
            emotion_connList.append(df_pairs['conn'][header])
            header += 1
        discourse_connLists.append(emotion_connList)
    connLists.append(discourse_connLists)

df_discourse['conn'] = connLists
df_discourse.to_csv('../data/discourse_withconn.csv', index=False)
