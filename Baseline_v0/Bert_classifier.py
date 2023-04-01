import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

class PairsWithConnectivesDataSet(Dataset):
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        self.data_path = r"data/pairs_withconn.csv"
        self.pairs_list = []
        with open(self.data_path,'r') as f:
            for line in f:
                line = line.split(',')
                emotion_clause,cause_candidate,conn,correctness,is_cause_conn = line[0],line[1],line[2],line[-2],line[-1]
                my_tuple = (emotion_clause,cause_candidate,conn,correctness,is_cause_conn)
                self.pairs_list.append(my_tuple)
        self.pairs_list.pop(0)

    def __getitem__(self, item):
        '''

        :param item:
        :return: tokenzied(text1,conn+text2,pairs_len,cottectness,is_cause_conn)
        '''
        line = self.pairs_list[item]
        # tokens = [ [mask] seg1 [sep] con seg2 [sep]
        # tokens = self.tokenizer(line[0], line[2]+line[1])
        correctness = 0 if line[3] == 'False' else 1
        is_cause_conn = 0 if line[4] == 'false' else 1

        encoding = self.tokenizer(line[0],line[2]+' '+line[1], return_tensors='pt', padding='max_length', truncation=True
                                  # , max_length=self.max_length
                                  )
        return {'input_ids': encoding['input_ids'].squeeze(),
                'token_type_ids':encoding['token_type_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'correctness': correctness,
                'is_cause_conn':is_cause_conn
                }
    def __len__(self):
        return len(self.pairs_list)


def train(model, dataloader, optimizer, device):
    model.train()
    losses = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        correctness = batch['correctness'].to(device)
        is_cause_conn = batch['is_cause_conn'].to(device)

        optimizer.zero_grad()
        # 这里使用is_cause_conn为标签
        outputs = model(input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, labels=is_cause_conn)
        loss = outputs.loss
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print("Loss=", np.mean(losses))


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    pairs_with_connectives_dataset = PairsWithConnectivesDataSet(tokenizer)
    pairs_with_connectives_dataloader = DataLoader(dataset=pairs_with_connectives_dataset,shuffle=True,batch_size=4)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, pairs_with_connectives_dataloader, optimizer, device)
    #
    # # Replace with your own data
    # texts = ["This is a positive sentence.", "because This is a negative sentence."]
    # labels = [1, 0]
    # max_length = 128
    #
    # dataset = BinaryClassificationDataset(texts, labels, tokenizer, max_length)
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    #
    # optimizer = AdamW(model.parameters(), lr=2e-5)
    #
    # num_epochs = 3
    # for epoch in range(num_epochs):
    #     print(f"Epoch {epoch + 1}/{num_epochs}")
    #     train(model, dataloader, optimizer, device)