import torch
from torch.utils.data import DataLoader
import tqdm
from datasets import load_dataset
from torchvision.models import resnet50
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import pickle

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
  # load the pretrained ResNet50 model
  # freeze the weights of the pre-trained layers
  # modify the last layer to output 200 classes
    def __init__(self):
        super(Resnet50TinyImageNet, self).__init__()
        self.model = resnet50()
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 200)
    
    def forward(self, x):
        return self.model(x)
    
    def train(self, train_loader, criterion, optimizer, num_epochs=10):
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

            print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f}'.format(
                epoch+1, train_loss, train_acc))
            
            # save model
            torch.save(self.model.state_dict(), 'resnet50_ft.ckpt')


    def test(self, test_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        predictions = []

        for image, label in tqdm.tqdm(test_loader):
            image = image.to(device)
            label = label.to(device)

            outputs = self.model(image)
            _, prediction = torch.max(outputs, 1)
            predictions.append(prediction)

        return predictions

            
## ______________

# Hyperparameters

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

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
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.train(train_loader, criterion, optimizer, EPOCHS)

pred = model.test(val_loader)

with open('pred.pkl', 'wb') as f:
    pickle.dump(pred, f)


