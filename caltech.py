# Import Libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

model_path="caltech_model_train_Alex"

##turning on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

##store classes
files=os.listdir("objectsSplitted/train")
sequence={}
for x in files:
    sequence[int(x.split(".")[0])]=x.split(".")[1]
class_labels=[]
for x in range(len(files)):
    class_labels.append(sequence[x+1])

# Show Image
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
    
def val_accuracy():
    print("Calculating Accuracy")
    total=0
    correct=0
    x=0
    with torch.no_grad():
        for data in val_loader:
            #print(x)
            x=x+1
            if x%100==0:
                print(x,end=",")
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #dataiter = iter(train_loader)
            #images, labels = dataiter.next()
            #show images
            #imshow(torchvision.utils.make_grid(images.cpu()), [class_labels[x] for x in predicted])
            #for x in labels:
            #    print(class_labels[x])
        print('Accuracy of the network: %d/%d , %d %%' % (correct,total,(100 * correct / total)))



# Specify transforms using torchvision.transforms as transforms
# library
transformations = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(127),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder("objectsSplitted/train", transform = transformations)
val_set = datasets.ImageFolder("objectsSplitted/val", transform = transformations)
test_set = datasets.ImageFolder("objectsSplitted/test", transform = transformations)
# Put into a Dataloader using torch library
train_loader = torch.utils.data.DataLoader(train_set, batch_size=6, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=6, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=6, shuffle=True)
#dataiter = iter(train_loader)
#images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images), [class_labels[x] for x in labels])


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*60*60, 300)
        self.fc2 = nn.Linear(300, 257)
        
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*60*60)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
    
class AlexNet(nn.Module):
    def __init__(self, num_classes=257):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 8*8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 8 * 8)
        x = self.classifier(x)
        return x
    

    
#net = Net()
net=AlexNet()
#sending model to cuda
net=net.to(device)

if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path))
    net.eval()
    print("Model Loaded")

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.1,momentum=0.9)

correct = 0
total = 0
for epoch in range(30):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i%1500==0:
            torch.save(net.state_dict(), model_path)
            print("Model Saved in",model_path)
        if i%3000==0:
            val_accuracy()
            

print('Finished Training')








