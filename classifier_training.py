import string

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms  # contains a collection of transformations
import torch
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset  # For training data and data processing

# Hyper-parameters and some variables for data processing
batch = 128
lr = 0.01
input_size = (28, 28)
n_classes = 26
valid_percent = 0.2
early_return = 2
# Checking whether GPU is working
train_on_gpu = torch.cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')
save_model_path = 'model-output.pt'

# Loading dataset
test_set = pd.read_csv('./dataset/sign_mnist_test.csv')
train_set = pd.read_csv('./dataset/sign_mnist_train.csv')

# Converting dataset into numpy (split sample data and target at the same time)
test_t = test_set.label.to_numpy()
test_sam = test_set.loc[:, test_set.columns != 'label'].to_numpy()
test_sam = test_sam.reshape((test_sam.shape[0], 1, input_size[0], input_size[1]))
train_t = train_set.label.to_numpy()
train_sam = train_set.loc[:, train_set.columns != 'label'].to_numpy()
train_sam = train_sam.reshape((train_sam.shape[0], 1, input_size[0], input_size[1]))

# Now converting numpy to tensor and construct a TensorDataset for Pytorch training
train_data = TensorDataset(torch.tensor(train_sam, dtype=torch.float32),
                           torch.tensor(train_t, dtype=torch.long))
test_data = TensorDataset(torch.tensor(test_sam, dtype=torch.float32),
                          torch.tensor(test_t, dtype=torch.long))

# # Check the dataset image tensor in the dataset can be
# # reconstruct into a picture correspond to its label
# target_list = dict(enumerate(string.ascii_uppercase))
# img, label = train_data[1]
# img = np.reshape(img, (28, 28))
# print(img.shape, label)
# plt.imshow(img, cmap='gray')
# print('Label:', target_list[label.item()])

# Split train_data into training set and validation set
train_data_size = train_data.tensors[0].shape[0]  # number of rows in sample tensor of train_data
valid_size = int(train_data_size * valid_percent)
train_ds, val_ds = torch.utils.data.random_split(train_data,
                                                 [train_data_size - valid_size, valid_size])


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128 * 5 * 5, out_features=64, bias=True),
            nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Linear(in_features=64, out_features=n_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

# # Check output of each layer
# model = CNN()
# summary(model, (1, 28, 28), batch_size=batch)

def train_model(model, train_dataset, val_dataset, save_file_name=save_model_path, learning_rate=lr,
                batch_size=batch, num_epochs=10):
    train_ld = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_ld = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if train_on_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Training loop
    for epoch in range(num_epochs):
        print(f'Training Epoch {epoch}...\n')
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        model.train()
        for i, (imgs, labels) in enumerate(train_ld):
            # print(f'Data shape: {imgs.shape}\n')
            if train_on_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()
            # Clear gradients
            optimizer.zero_grad()
            output = model(imgs)


            # Calculating loss
            loss = criterion(output, labels)
            loss.backward()

            # Update param.
            optimizer.step()

            # Calculating train loss
            train_loss += loss.item() * imgs.size(0)

            # Calculating accuracy
            _, pred = torch.max(output, dim=1)
            correct = pred.eq(labels.data.view_as(pred))
            accuracy = torch.mean(correct.type(torch.FloatTensor))
            train_acc += accuracy.item() * imgs.size(0)

        with torch.no_grad():
            model.eval()

            for imgs, labels in val_ld:
                if train_on_gpu:
                    imgs, labels = imgs.cuda(), labels.cuda()

                # Forward pass
                output = model(imgs)

                # compute loss
                loss = criterion(output, labels)
                valid_loss += loss.item() * imgs.size(0)

                # Calculate validation accuracy
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(labels.data.view_as(pred))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples
                valid_acc += accuracy.item() * imgs.size(0)

            # Compute average loss and accuracy
            train_loss = train_loss / len(train_ld.dataset)
            valid_loss = valid_loss / len(val_ld.dataset)
            train_acc = train_acc / len(train_ld.dataset)
            valid_acc = valid_acc / len(val_ld.dataset)

            print(f'Epoch: {epoch} \tTrain Loss: {train_loss:.4f} '
                  f'\tValidation Loss: {valid_loss:.4f}')
            print(
                f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t '
                f'Validation Accuracy: {100 * valid_acc:.2f}%\n'
            )

            # Early return detection
            if valid_loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_file_name)
                valid_loss_min = valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_return:
                    print(f'Early return at epoch {epoch}!\n')
                    break

    return model

cnn_model = CNN()
cnn_model = train_model(cnn_model, train_ds, val_ds)

# TODO: Create function for loading and saving model

# def main():
#     return
#
#
# if __name__ == "__main__":
#     main()
