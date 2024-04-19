import string
import cv2
import numpy as np
import pandas as pd
import os
# Visualizations
import matplotlib.pyplot as plt
import torchvision.transforms as transforms  # contains a collection of transformations
import torch
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset  # For training data and data processing
from cvzone.HandTrackingModule import HandDetector

# Hyper-parameters and some variables for data processing
batch = 128
lr = 0.01
input_size = (28, 28)
n_classes = 26
valid_percent = 0.2

# Checking whether GPU is working
train_on_gpu = torch.cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')
save_model_path = 'model-output.pt'
checkpoint_path = 'checkpoint.pth'

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

image_transform = transforms.Compose([
            # transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
            # transforms.RandomRotation(degrees=10),
            # transforms.ColorJitter(brightness=0.25),
            transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(size=28),  # Image net standards
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406],
            #                      [0.229, 0.224, 0.225])  # Imagenet standards
        ])


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=32 * 5 * 5, out_features=128, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # nn.Dropout(p=0.5, inplace=False),
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

def accuracy(model, dataset):
    dl = DataLoader(dataset, batch_size=batch, shuffle=False)
    total_acc = 0
    with torch.no_grad():
        model.eval()

        for imgs, labels in dl:
            if train_on_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            # Forward pass
            output = model(imgs)

            # Calculate validation accuracy
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(labels.data.view_as(pred))
            acc = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples
            total_acc += acc.item() * imgs.size(0)
        total_acc = total_acc / len(dl.dataset)
        # print(len(dl.dataset))
        print(f'Test acc: {100 * total_acc:.2f}%.\n')
        return

def train_model(model, train_dataset, val_dataset, save_file_name=save_model_path, learning_rate=lr,
                batch_size=batch, num_epochs=20, early_return=3):
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
            # imgs = image_transform(imgs)
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
            acc = torch.mean(correct.type(torch.FloatTensor))
            train_acc += acc.item() * imgs.size(0)

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
                acc = torch.mean(correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples
                valid_acc += acc.item() * imgs.size(0)

            # Compute average loss and accuracy
            train_loss = train_loss / len(train_ld.dataset)
            valid_loss = valid_loss / len(val_ld.dataset)
            train_acc = train_acc / len(train_ld.dataset)
            valid_acc = valid_acc / len(val_ld.dataset)

            # Record the result of this epoch
            history.append([train_loss, valid_loss, train_acc, valid_acc])

            print(f'Epoch: {epoch} \tTrain Loss: {train_loss:.4f} '
                  f'\tValidation Loss: {valid_loss:.4f}')
            print(
                f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t '
                f'Validation Accuracy: {100 * valid_acc:.2f}%\n'
            )
            accuracy(model, test_data)

            # Early return detection
            if valid_loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_file_name)
                valid_loss_min = valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

                if epochs_no_improve >= early_return:
                    print(f'Early return at epoch {epoch} with lowest loss {valid_loss_min}!\n')
                    # Early return, then reload the best state dict
                    model.load_state_dict(torch.load(save_file_name))
                    # Record the state of optimizer which helps to save the model
                    model.optimizer = optimizer
                    # Re-construct history to a Dataframe which easy to plot
                    history = pd.DataFrame(history,
                                           columns=['train_loss', 'valid_loss',
                                                    'train_acc', 'valid_acc'])
                    return model, history
    # Record the state of optimizer which helps to save the model
    model.optimizer = optimizer
    # Re-construct history to a Dataframe which easy to plot
    history = pd.DataFrame(history,
                           columns=['train_loss', 'valid_loss',
                                    'train_acc', 'valid_acc'])
    return model, history


def plot_history(history):
    # plt.figure(figsize=(8, 6))
    figure, axis = plt.subplots(1, 2)
    for c in ['train_loss', 'valid_loss']:
        axis[0].plot(
            history[c], label=c)
    axis[0].legend()
    # axis[0].xlabel('Epoch')
    # axis[0].ylabel('Average Negative Log Likelihood')
    axis[0].set_title('Training and Validation Losses')

    for c in ['train_acc', 'valid_acc']:
        axis[1].plot(
            100 * history[c], label=c)
    axis[1].legend()
    # axis[1].xlabel('Epoch')
    # axis[1].ylabel('Average Negative Log Likelihood')
    axis[1].set_title('Training and Validation Accuracy')

    plt.show()

def video_capture():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    offset = 20
    model = CNN()
    if train_on_gpu:
        model.load_state_dict(torch.load('./model/best-model.pt'))
    else:
        model.load_state_dict(torch.load('./model/best-model.pt',
                              map_location=torch.device('cpu')))
    while True:
        success, img = cap.read()
        hand_img = img.copy()
        hands, hand_img = detector.findHands(hand_img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            x_offset = x + offset
            y_offset = y + offset
            # Crop image
            # Checking corner cases and adapt them
            if x - offset < 0:
                x = 0
            else:
                x = x - offset

            if y - offset < 0:
                y = 0
            else:
                y = y - offset
            img_cropped = img[y:y_offset + h, x:x_offset + w]
            img_cropped = cv2.resize(img_cropped, (28, 28), interpolation=cv2.INTER_AREA)
            cv2.imshow("ImgCropped", img_cropped)
            img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY).reshape((1, 1, 28, 28))
            print(img_cropped.shape)
            # sample = img_cropped.reshape((1, 28, 28))
            sample = torch.from_numpy(img_cropped)
            sample = sample.type(torch.float32)
            with torch.no_grad():
                if train_on_gpu:
                    print(model(sample.cuda()))
                else:
                    output = model(sample)
                    a, b = torch.max(output, dim=1)
                    print(output)
                    print(a, b)
            # cv2.imshow("ImgCropped", img_cropped)
        cv2.imshow("Image", hand_img)
        cv2.waitKey(1)


# def save_model(model, path):
#     checkpoint = {'state_dict': model.state_dict(), 'optimizer': model.optimizer.state_dict()}
#
#     return
#
# def load_model(path):
#     return

cnn_model = CNN()
# cnn_model.load_state_dict(torch.load('./model/model-output-0.1255.pt'))
# accuracy(cnn_model, test_data)
# test_dl = DataLoader(test_data, batch_size=batch, shuffle=False)
# for imgs, labels in test_dl:
#     output = cnn_model(imgs)

# video_capture()
cnn_model, history = train_model(cnn_model, train_ds, val_ds)

# TODO: Create function for loading and saving model
# TODO: Implement a function for video capture and interfere the hand sign

# def main():
#     return
#
#
# if __name__ == "__main__":
#     main()
