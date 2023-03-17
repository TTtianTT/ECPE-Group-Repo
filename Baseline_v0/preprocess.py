import csv

# With gold emotion labels
# Set pairs

# Init param
cause_conn = []  # Set cause_conn

# Load connectives info
with open ('cause_conn_modified.txt', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        for word in line.split(','):
            cause_conn.append(word)
        line = f.readline()

# Init csv for recording result
with open ('pairs.csv', 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['emotion_clause', 'cause_candidate', 'pairs_len', 'correctness'])

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
        
        # Delete original connectives
        for i in range(length):

            # Delete bigram first
            del_pos = []
            new_content = ''
            for j in range(len(refined_content[i]) - 1):
                if refined_content[i][j:j+2] in cause_conn:
                    del_pos.extend([j,j+1])
            for j in range(len(refined_content[i])):
                if j not in del_pos:
                    new_content = new_content + refined_content[i][j]
            refined_content[i] = new_content

            # Delete unigram later
            del_pos = []
            new_content = ''
            for j in range(len(refined_content[i])):
                if refined_content[i][j] in cause_conn:
                    del_pos.append(j)
            for j in range(len(refined_content[i])):
                if j not in del_pos:
                    new_content = new_content + refined_content[i][j]
            refined_content[i] = new_content

        # Get emo_index and cau_index
        emo_index = []
        cau_index = []
        for pair in pairs_index:
            if pair[0] not in emo_index:
                emo_index.append(pair[0])  # pair[0] = emo_index
                cau_index.append([])
            cau_index[emo_index.index(pair[0])].append(pair[1])  # pair[1] = cau_index

        # For each emo_clause
        # Construct paris
        pairs = ''
        pairs_len = ''
        correctness = ''
        for emo_clause_index in emo_index:
            for i in range(length):
                if i + 1 not in emo_index:
                    # pairs = '[CLS]' + refined_content[emo_clause_index - 1] + '[SEP]' + '[MASK]' + refined_content[i] + '[SEP]'
                    emotion_clause = refined_content[emo_clause_index - 1]
                    cause_candidate = refined_content[i]
                    pairs_len = str(len(refined_content[emo_clause_index - 1])) + ',' + str(len(refined_content[i]))
                    correctness = 'false'
                    if i + 1 in cau_index[emo_index.index(emo_clause_index)]:
                        correctness = 'true'
                    
                    # Write result in csv
                    with open ('pairs.csv', 'a', encoding='utf-8', newline='') as g:
                        csv_writer = csv.writer(g)
                        csv_writer.writerow([emotion_clause, cause_candidate, pairs_len, correctness])
            
        sec = f.readline()  # Read following section length