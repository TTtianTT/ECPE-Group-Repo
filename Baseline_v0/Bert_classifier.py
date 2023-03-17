import torch
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from transformers import BertModel, BertTokenizer, BertForSequenceClassification

# Load dataset
df = pd.read_csv('pairs_withconn.csv')

# Designate device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# Preprocess
labels = {'False':0,'True':1}
pairs = []
for i in range(len(df)):
    emotion_clause = df['emotion_clause'][i]
    cause_candidate = df['cause_candidate'][i]
    # pairs.append('[CLS]' + str(emotion_clause) + '[SEP]' + '[MASK]' + str(cause_candidate) + '[SEP]')
    texts = '[CLS]' + str(emotion_clause) + str(cause_candidate) + '[SEP]'

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = df['correctness'].map(labels)
        self.texts = [tokenizer(text, 
                                padding='max_length', 
                                max_length = 512, 
                                truncation=True,
                                return_tensors="pt") 
                      for text in texts]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)
    # Use DataLoader
    # Get data with standard batch_size
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    # Init loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    # Train the model
    for epoch_num in range(epochs):
      # Init param
            total_acc_train = 0
            total_loss_train = 0
      # tqdm for view
            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
        # Get output
                output = model(input_id, mask)
                # Calculate loss
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                # Calculate acc
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
        # Model update
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            # ------ Validate model -----------
            # Init param
            total_acc_val = 0
            total_loss_val = 0
      # No grad for validation
            with torch.no_grad():
                # Get validate dataset and validate the model
                for val_input, val_label in val_dataloader:
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
  
                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')

def evaluate(model, test_data):

    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              output = model(input_id, mask)
              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc   
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


np.random.seed(0)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=0), 
                                     [int(.8*len(df)), int(.9*len(df))])
EPOCHS = 5
LR = 1e-6
train(model, df_train, df_val, LR, EPOCHS)
evaluate(model, df_test)