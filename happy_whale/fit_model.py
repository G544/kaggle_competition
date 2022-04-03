from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm, tqdm_notebook
import numpy as np
import pickle
import torch


class Model():
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else  torch.device("cpu")
    
    
    def __init__(self, model, opt, criterion, scheduler, epochs=25, batch_size=8):
        self.model = model.to(self.DEVICE)
        self.opt = opt
        self.criterion = criterion
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
        
    def fit(self, train_dataset, val_dataset):
        self.train(train_dataset, val_dataset, self.model, self.epochs, self.batch_size, self.opt, self.criterion, self.scheduler)
        print('Training of model have been finished')
            
    def train(self, train_dataset, val_dataset, model, epochs, batch_size,opt, criterion, scheduler):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        history = []
        log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
        val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

        with tqdm(desc="epoch", total=epochs) as pbar_outer:
            #opt = torch.optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                train_loss, train_acc = self._fit_epoch(model, train_loader, criterion, opt, scheduler)
                print("loss", train_loss)

                val_loss, val_acc = self._eval_epoch(model, val_loader, criterion)
                history.append((train_loss, train_acc, val_loss, val_acc.to('cpu')))

                pbar_outer.update(1)
                tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss,\
                                               v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))

        return history

        
    
    def _fit_epoch(self, model, train_loader, criterion, optimizer, scheduler):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        processed_data = 0
        scheduler.step()

        for inputs, labels in train_loader:
            inputs = inputs.to(self.DEVICE)
            labels = labels.to(self.DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            processed_data += inputs.size(0)

        train_loss = running_loss / processed_data
        train_acc = running_corrects.cpu().numpy() / processed_data # не забыть отправить обратно на cpu
        return train_loss, train_acc


    def _eval_epoch(self, model, val_loader, criterion):
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        processed_size = 0

        for inputs, labels in val_loader:
            inputs = inputs.to(self.DEVICE)
            labels = labels.to(self.DEVICE)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            processed_size += inputs.size(0)
        val_loss = running_loss / processed_size
        val_acc = running_corrects.double() / processed_size
        return val_loss, val_acc
    
    def predict(self, test_dataset):
        return self._predict(self.model, test_dataset)



    def _predict(self, model, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            logits = []

            for inputs in test_loader:
                inputs = inputs.to(self.DEVICE)
                model.eval()
                outputs = model(inputs).cpu()
                logits.append(outputs)

        probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
        y_pred = np.argmax(probs,-1)
        print(y_pred)
        preds_class = [self.label_encoder.classes_[i] for i in y_pred]
        return preds_class
