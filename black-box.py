import torch
from torch.utils.data import DataLoader
import tqdm
from datasets import load_dataset
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import pickle
from sklearn.metrics import accuracy_score


class TinyImageNetDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image = self.dataset[index]["image"]
        label = self.dataset[index]["label"]

        if self.transform:
            image = self.transform(image)
        # if grayscale, convert to 3-channel
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)
            
        label = torch.tensor(label)
        return image, label

    def __len__(self):
        return len(self.dataset)
    
class Resnet50TinyImageNet(nn.Module):
    def __init__(self):
        super(Resnet50TinyImageNet, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 200)
    
    def forward(self, x):
        return self.model(x)
    
    def train(self, train_loader, val_loader, criterion, optimizer, num_epochs=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        for epoch in range(num_epochs):
            train_loss = 0.0
            train_acc = 0.0
            
            self.model.train()
            for image, label in tqdm.tqdm(train_loader):
                image = image.to(device)
                label = label.to(device)
                
                optimizer.zero_grad()
                
                outputs = self.model(image)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * image.size(0)
                _, prediction = torch.max(outputs, 1)
                train_acc += torch.sum(prediction == label.data)
            
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_acc / len(train_loader.dataset)

            self.model.eval()
            val_loss = 0.0
            val_acc = 0.0

            for image, label in tqdm.tqdm(val_loader):
                image = image.to(device)
                label = label.to(device)

                outputs = self.model(image)
                loss = criterion(outputs, label)

                val_loss += loss.item() * image.size(0)
                _, prediction = torch.max(outputs, 1)
                val_acc += torch.sum(prediction == label.data)

            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_acc / len(val_loader.dataset)

            print("Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}".format(epoch+1, train_loss, train_acc, val_loss, val_acc))


    def test(self, test_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        predictions = []
        true = []

        for image, label in tqdm.tqdm(test_loader):
            image = image.to(device)
            label = label.to(device)

            outputs = self.model(image)
            _, prediction = torch.max(outputs, 1)
            predictions.append(prediction)
            true.append(label.data)

        predictions = torch.cat(predictions, dim=0)
        true = torch.cat(true, dim=0)

        print("Accuracy: ", accuracy_score(true.cpu(), predictions.cpu()))

        return predictions
            
## ______________

# Hyperparameters

BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001
MOMENTUM = 0.9

## ______________

# Load dataset

dataset = load_dataset("Maysee/tiny-imagenet")
train = dataset["train"]
val = dataset["valid"]

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = TinyImageNetDataset(train, transform)
val_dataset = TinyImageNetDataset(val, transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

## ______________

# Train model

model = Resnet50TinyImageNet()

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

model.train(train_loader, val_loader, criterion, optimizer, EPOCHS)

torch.save(model.state_dict(), './model/res15.pth')

## ______________

# Test model

# pred = model.test(val_loader)

# with open('pred_15.pkl', 'wb') as f:
#     pickle.dump(pred, f)



