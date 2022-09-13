from torch import nn
import torch
import tqdm
from src.components.model import NeuralNet




class Trainer: 
    def __init__(self): 
        self.trainLoader = None
        self.testLoader = None

    def train(self):
        net = NeuralNet()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        criterion = nn.CrossEntropyLoss()

        # optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)

        epochs = 5

        train_loss , train_accuracy = [], []

        net.train()

        for epoch in range(epochs):
            print('Training')
            running_loss = 0.0
            running_correct = 0

            for data in tqdm(self.trainLoader):

                data, target = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = net(data)

                loss = criterion(outputs, target)

                running_loss += loss.item()

                _, preds = torch.max(outputs.data, 1)

                running_correct += (preds == target).sum().item()

                loss.backward()
                optimizer.step()
                
            loss = running_loss/len(self.trainLoader.dataset)
            accuracy = 100. * running_correct/len(self.trainLoader.dataset)
            
            print(f"Train Loss: {loss:.4f}, Train Acc: {accuracy:.2f}")

    def test(self):
        total = 0
        running_correct = 0
        test_loss = 0

        model.eval()

        with torch.no_grad():
        for data in tqdm(valLoader):
            total +=1
            img = data[0].to(device)
            label = data[1].to(device)
            
            pred = net(img)
            loss = criterion(pred,label)

            _, preds = torch.max(pred.data, 1)
            running_correct += (preds == label).sum().item()
            test_loss += loss.item()