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
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
import numpy as np

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


    def test(self, test_loader, device):

        self.model.eval()
        predictions = []
        true = []
        images = []

        for image, label in tqdm.tqdm(test_loader):
            image = image.to(device)
            label = label.to(device)

            outputs = self.model(image)
            _, prediction = torch.max(outputs, 1)
            predictions.append(prediction)
            true.append(label.data)
            images.append(image)

        predictions = torch.cat(predictions, dim=0)
        true = torch.cat(true, dim=0)
        images = torch.cat(images, dim=0)

        print("Accuracy: ", accuracy_score(true.cpu(), predictions.cpu()))

        return predictions, true, images

def SIMBA_single_unt(x,y,model,epsilon=0.2,num_iters = 5000):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  n_dims = x.view(1,-1).size(1)
  perm = torch.randperm(n_dims)
  x = x.unsqueeze(0)
  x_probs = F.softmax(model(x.to(device)))
  def_prob = x_probs[0][y]

  
  for i in range(num_iters):
    delta = torch.zeros(n_dims)
    delta[perm[i]] = epsilon
    add_vec = x + delta.view(x.size())
    probs = F.softmax(model(add_vec.to(device)))
    new_prob = probs[0][y]


    if new_prob < def_prob:
      x = add_vec
      def_prob = new_prob
    else:
      x = x - delta.view(x.size())
      # def_prob = F.softmax(model(x))[0][y]
      probs = F.softmax(model(x.to(device)))
      def_prob = probs[0][y]

    ad_prob, ad_pred = torch.max(probs, 1)

    x = x.to(torch.device("cpu"))
    ad_pred = ad_pred.to(torch.device("cpu"))
    ad_prob = ad_prob.to(torch.device("cpu"))
    x_probs = x_probs.to(torch.device("cpu"))
    probs = probs.to(torch.device("cpu"))
    y = y.to(torch.device("cpu"))

    if ad_pred[0].item()!=y:
      break

  return x.squeeze(), i+1, ad_pred[0].item(),x_probs[0][y],probs[0][y],x_probs[0][ad_pred],ad_prob

def SIMBA(true_preds, images, true, model, eps=0.2):
    adv_img = []
    iter = []
    diff_init_pred = []
    diff_final_pred = []
    new_class = []

    for i in tqdm.tqdm(range(len(true_preds))):
        adv_img_,iter_,new_class_,init_prob_act_,fin_prob_act_,init_prob_adv_,fin_prob_adv_ = SIMBA_single_unt(images[true_preds[i]], true[true_preds[i]], model, eps)
        adv_img.append(adv_img_)
        iter.append(iter_)
        new_class.append(new_class_)

        init_prob_act_ = init_prob_act_.detach().numpy()
        fin_prob_act_ = fin_prob_act_.detach().numpy()
        init_prob_adv_ = init_prob_adv_.detach().numpy()
        fin_prob_adv_ = fin_prob_adv_.detach().numpy()


        diff_init_pred.append(init_prob_act_-init_prob_adv_)
        diff_final_pred.append(fin_prob_act_-fin_prob_adv_)

    avg_iter = np.mean(iter)

    avg_diff_init_pred = np.mean(diff_init_pred)
    avg_diff_final_pred = np.mean(diff_final_pred)



    return adv_img, iter, new_class, avg_iter, avg_diff_init_pred, avg_diff_final_pred


# ________

# Hyperparameters

BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001
MOMENTUM = 0.9

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

# Train model

model = Resnet50TinyImageNet()

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# model.train(train_loader, val_loader, criterion, optimizer, EPOCHS)

# torch.save(model.state_dict(), './model/res3.pth')

## ______________

# Test model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load('./model/res3.pth',  map_location=device))

pred, true, images = model.test(val_loader, device)

images = images.to(torch.device("cpu"))
true = true.to(torch.device("cpu"))
pred = pred.to(torch.device("cpu"))
model = model.to(torch.device("cpu"))

true_preds = []
for i in range(len(pred)):
    if pred[i] == true[i]:
        true_preds.append(i)


adv_img2, iter2, new_class2, avg_iter2, avg_diff_init_pred2, avg_diff_final_pred2 = SIMBA(true_preds[:500], images, true, model, eps=0.2)

# save as pickle file
with open('./adv_img2.pkl', 'wb') as f:
    pickle.dump(adv_img2, f)

with open('./iter2.pkl', 'wb') as f:
    pickle.dump(iter2, f)

with open('./new_class2.pkl', 'wb') as f:
    pickle.dump(new_class2, f)

with open('./avg_iter2.pkl', 'wb') as f:
    pickle.dump(avg_iter2, f)

with open('./avg_diff_init_pred2.pkl', 'wb') as f:
    pickle.dump(avg_diff_init_pred2, f)

with open('./avg_diff_final_pred2.pkl', 'wb') as f:
    pickle.dump(avg_diff_final_pred2, f)
