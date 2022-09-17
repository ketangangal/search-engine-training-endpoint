from torch import nn
import torch
import numpy as np 
from src.components.model import NeuralNet
from src.components.data_preprocessing import DataPreprocessing
import time 

class Trainer: 
    def __init__(self,trainLoader,testLoader,validLoader): 
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.validLoader = validLoader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = nn.CrossEntropyLoss()
        self.model = NeuralNet(101).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)
        self.evaluation = True


    def train_model(self,epochs):
      print("Start training...\n")
      for epoch_i in range(epochs):
          print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
          print("-"*70)

          # Measure the elapsed time of each epoch
          t0_epoch, t0_batch = time.time(), time.time()

          # Reset tracking variables at the beginning of each epoch
          total_loss, batch_loss, batch_counts = 0, 0, 0

          # Put the model into the training mode
          self.model.train()

          for step, batch in enumerate(self.trainLoader):
            batch_counts +=1
            # Load batch to GPU
            img = batch[0].to(self.device)
            labels  = batch[1].to(self.device)

            # Zero out any previously calculated gradients
            self.model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = self.model(img)

            # Compute loss and accumulate the loss values
            loss = self.criterion(logits, labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Update parameters and the learning rate
            self.optimizer.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(self.trainLoader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
          avg_train_loss = total_loss / len(self.trainLoader)

          print("-"*70)
          # =======================================
          #               Evaluation
          # =======================================
          if self.evaluation == True:
              # After the completion of each training epoch, measure the model's performance
              # on our validation set.
              val_loss, val_accuracy = self.evaluate()

              # Print performance over the entire training data
              time_elapsed = time.time() - t0_epoch
              
              print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
              print("-"*70)
          print("\n")
        
      print("Training complete!")


    def evaluate(self, validate = False):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        self.model.eval()

        # Tracking variables
        val_accuracy = []
        val_loss = []

        loader = self.testLoader if not validate else self.validLoader

        for batch in loader:
            # Load batch to GPU
            img = batch[0].to(self.device)
            labels  = batch[1].to(self.device)
            # Compute logits
            with torch.no_grad():
                logits = self.model(img)

            # Compute loss
            loss = self.criterion(logits, labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = (preds == labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy

    def save_model_in_pth(self):
        pass

    def save_model_int_onnx(self):
        pass

if __name__ == "__main__":
    dp = DataPreprocessing()
    loaders = dp.run_step()
    trainer = Trainer(loaders["train_data_loader"][0],loaders["test_data_loader"][0],loaders["valid_data_loader"][0])
