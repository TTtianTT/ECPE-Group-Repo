'''
We use the following example for explaination.

4 12
 (12,9), (12,10), (12,11)
1,null,null,为 尽快 将 女子 救 下
2,null,null,指挥员 立即 制订 了 救援 方案
3,null,null,第一组 在 楼下 铺设 救生 气垫
4,null,null,并 对 周围 无关 人员 进行 疏散
5,null,null,另一组 队员 快速 爬 上 6 楼
6,null,null,在 楼 内 对 女子 进行 劝说
7,null,null,劝说 过程 中
8,null,null,消防官兵 了解 到
9,null,null,该 女子 是 由于 对方 拖欠 工程款
10,null,null,家中 又 急需 用钱
11,null,null,生活 压力 大
12,sadness,无奈,无奈 才 选择 跳楼 轻生

'''

import csv

# Define class
class Statistics():
    def __init__(self, key, emo_conn, cau_conn, type, dis):
        self.key = key
        self.emo_conn = emo_conn
        self.cau_conn = cau_conn
        self.type = type
        self.dis = dis
        self.frequency = 0

# Init param
result = []  # Statistic result
key = []  # Primary key for index
conn_words = []  # Set connectives

# Load cause connectives
with open ('cause_conn.txt', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        for word in line.split(','):
            conn_words.append(word)
        line = f.readline()

with open ('all_data_pair.txt', 'r', encoding='utf-8') as f:  # Encode by utf-8 for Chinese
    sec = f.readline()  # Read section ID and length

    # For each section
    while sec:

        # Get section length
        num = sec.split(' ')
        length = int(num[1])
        content = ['' for i in range(length)]
        refined_content = ['' for i in range(length)]

        pairs = f.readline().lstrip().rstrip()  # Get the index of pairs and delete the beginning ' ' and ending '\n'

        # Get the index of pairs (int)
        pairs_index = []
        for pair in pairs.split(', '):
            pairs_index.append(list(map(int, pair.lstrip('(').rstrip(')').split(','))))

        # Get the content of section
        for i in range(length):
            content[i] = f.readline().lstrip().rstrip().split(',')[3]

            # Get the raw content
            for word in content[i].split(' '):
                refined_content[i] += word
        
        # For each pair
        for pair in pairs_index:
            dis = pair[1] - pair[0]  # Calculate dis = cause - emotion

            # Emotion clause
            emo_conn_flag = 0  # Set no conn as default
            emo_conn = ''
            # Judge double-character word later to cover the result as long as possible
            for i in range(len(refined_content[pair[1] - 1])):  # Single-character word
                if refined_content[pair[1] - 1][i] in conn_words:
                    emo_conn_flag = 1
                    emo_conn = refined_content[pair[1] - 1][i]
            for i in range(len(refined_content[pair[1] - 1]) - 1):  # Double-character word
                possible_conn = refined_content[pair[1] - 1][i] + refined_content[pair[1] - 1][i + 1]
                if possible_conn in conn_words:
                    emo_conn_flag = 1
                    emo_conn = possible_conn
            
            # Cause clause
            cau_conn_flag = 0  # Set no conn as default
            cau_conn = ''
            # Judge double-character word later to cover the result as long as possible
            for i in range(len(refined_content[pair[0] - 1])):  # Single-character word
                if refined_content[pair[0] - 1][i] in conn_words:
                    cau_conn_flag = 1
                    cau_conn = refined_content[pair[0] - 1][i]
            for i in range(len(refined_content[pair[0] - 1]) - 1):  # Double-character word
                possible_conn = refined_content[pair[0] - 1][i] + refined_content[pair[0] - 1][i + 1]
                if possible_conn in conn_words:
                    cau_conn_flag = 1
                    cau_conn = possible_conn

            # Judge structure type
            # type0 (emo, cau), type1 (emo, conn, cau), type2 (conn, emo, cau), type3 (conn, emo, conn, cau)
            # We always rewrite the sentence to make sure emo is ahead of cau for our research
            type = 0
            if cau_conn_flag == 1:
                type = 1
            if emo_conn_flag == 1:
                type = 2
            if cau_conn_flag == 1 & emo_conn_flag == 1:
                type = 3

            # Get statistics
            pair_key = emo_conn + cau_conn + str(type) + str(dis)  # Set unique key

            if pair_key not in key:
                key.append(pair_key)
                new_situation = Statistics(pair_key, emo_conn, cau_conn, type, dis)
                result.append(new_situation)
            
            result[key.index(pair_key)].frequency += 1

        sec = f.readline()  # Read following section length

# Write result in csv
with open ('result.csv', 'w', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)

    csv_writer.writerow(['pair_key', 'emo_conn', 'cau_conn', 'type', 'dis', 'frequency'])
    for item in result:
        csv_writer.writerow([item.key, item.emo_conn, item.cau_conn, item.type, item.dis, item.frequency])