import numpy as np
import csv

# Define class
class Statistics():
    def __init__(self, key, conn, conn_head_sem_class):
        self.key = key  # conn + conn_head_sem_class
        self.conn = conn
        self.conn_head_sem_class = conn_head_sem_class
        self.frequency = 0

# Init param
result = []  # Statistic result
keylist = []  # Primary key for index
conn_words = []  # Set connectives
conn_words_count = []  # Set connectives count

# Get statistics
def run_statistics(conn, conn_head_sem_class):
    key = conn + '+' + conn_head_sem_class  # Set unique key
    if key not in keylist:
        keylist.append(key)
        if conn not in conn_words:
            conn_words.append(conn)
            conn_words_count.append(0)
        new_situation = Statistics(key, conn, conn_head_sem_class)
        result.append(new_situation)
    conn_words_count[conn_words.index(conn)] += 1
    result[keylist.index(key)].frequency += 1

# Load dataset
with open ('pdtb2.csv', 'r', newline='') as f:
    csv_reader = csv.reader(f,delimiter=',')
    # For each instance (row)
    for row in csv_reader:
        if row[0] == 'Relation':
            continue
        # ConnHead
        if row[9] != '':
            conn = row[9]
            for index in range(12,14):
                if row[index] != '':
                    conn_head_sem_class = row[index]
                    run_statistics(conn, conn_head_sem_class)
        # Conn1
        if row[10] != '':
            conn = row[10]
            for index in range(12,14):
                if row[index] != '':
                    conn_head_sem_class = row[index]
                    run_statistics(conn, conn_head_sem_class)
        # Conn2
        if row[11] != '':
            conn = row[10]
            for index in range(14,16):
                if row[index] != '':
                    conn_head_sem_class = row[index]
                    run_statistics(conn, conn_head_sem_class)

# Write result in csv
with open ('result.csv', 'w', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)

    csv_writer.writerow(['key', 'conn', 'conn_head_sem_class', 'frequency', 'conn_total_count'])
    for item in result:
        csv_writer.writerow([item.key, item.conn, item.conn_head_sem_class, item.frequency, conn_words_count[conn_words.index(item.conn)]])
