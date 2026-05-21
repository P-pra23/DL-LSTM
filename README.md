# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
<img width="732" height="767" alt="image" src="https://github.com/user-attachments/assets/9f97d35c-9e33-4f6d-8897-f2fc4f57ca98" />


## DESIGN STEPS
## STEP 1:

Import the required libraries and load the NER dataset.

## STEP 2:

Preprocess the text data and convert words and tags into numerical indices.

## STEP 3:

Split the dataset into training and testing sets and create DataLoaders.

## STEP 4:

Build the BiLSTM model with embedding, LSTM, dropout, and linear layers.

## STEP 5:

Train the model using CrossEntropyLoss and an optimizer.

## STEP 6:

Evaluate the model performance and predict named entities on test data.



## PROGRAM

### Name:Prathikshaa

### Register Number:212224100043

```python
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size,
                 embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return self.fc(x)

model = BiLSTMTagger(
    len(word2idx) + 1,
    len(tag2idx)
).to(device)

loss_fn = nn.CrossEntropyLoss()

        


model=BiLSTMTagger(len(word2idx)+1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses,val_losses=[],[]
    for epoch in range(epochs):
      model.train()
      total_loss=0
      for batch in train_loader:
        input_ids=batch["input_ids"].to(device)
        labels=batch["labels"].to(device)
        optimizer.zero_grad()
        outputs=model(input_ids)
        loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
      train_losses.append(total_loss)
      model.eval()
      val_loss=0
      with torch.no_grad():
        for batch in test_loader:
          input_ids=batch["input_ids"].to(device)
          labels=batch["labels"].to(device)
          outputs=model(input_ids)
          loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
          val_loss+=loss.item()
      val_losses.append(val_loss)
      print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f},Val Loss={val_loss:.4f}")

    return train_losses, val_losses

```

### OUTPUT

## Loss Vs Epoch Plot

<img width="1167" height="626" alt="image" src="https://github.com/user-attachments/assets/fa50eb84-3796-45f1-bb5f-66dae0dbc5ce" />


### Sample Text Prediction
<img width="732" height="517" alt="image" src="https://github.com/user-attachments/assets/cccce57b-a9e7-48f0-8a7d-c29c3fb0e7cc" />


## RESULT


The LSTM-based Named Entity Recognition (NER) model was successfully developed and trained using the dataset. The BiLSTM model effectively identified named entities such as person names, locations, and organizations from the input text. The training and testing processes were completed successfully, and the model achieved good accuracy in predicting entity tags for unseen sentences.
