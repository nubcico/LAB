import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import timm
import os
import cv2
import numpy as np
import gc
from PIL import Image

# asssss hello heelli wiwoudlkejlajheoihoieaho
print("hello")
def sort_cmp (key):
    ans = 0
    for c in key:
        if (c.isdigit() == True):
            ans = ans * 10 + int (c)
    return ans
# Define the ViT model
class ViTEmotionClassifier(nn.Module):
    def __init__(self, num_classes, img_size=224):
        super(ViTEmotionClassifier, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

subfolder="subject2"
folder_path = f"C:\\Users\\user.DESKTOP-HI4HHBR\\Desktop\\frames(1)\\{subfolder}\\"
label_file_path = f"C:\\Users\\user.DESKTOP-HI4HHBR\\Desktop\\labels\\{subfolder}_labels.txt"
datav = []
labelsv = []
emotions = ["Anger", "Neutral", "Sadness", "Calmness", "Happiness"]

subfolder="subject2"

label_file_path = f"C:\\Users\\user.DESKTOP-HI4HHBR\\Desktop\\labels\\{subfolder}_labels.txt"

labelsv = []  
with open(label_file_path, 'r') as file:
    lines = file.readlines()[1:]

for line in lines:
    for label in line.strip(): 
        label=int(label)
        labelsv.append(label)
# Assuming labels is your list of labels
# Set your folder path
folder_path = f"C:\\Users\\user.DESKTOP-HI4HHBR\\Desktop\\frames(1)\\{subfolder}\\"

# Create file_labels by pairing each label with its corresponding file name
file_labels = [(f"frame_{i+1}.png", label) for i, label in enumerate(labelsv)]

for file_name, label in file_labels:
    file_path = os.path.join(folder_path, file_name)
    frame = cv2.imread(file_path)
    if frame is not None:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        #print("pil_frame:")
        #print(img)
        resized_frame = transform(pil_img)
        #print("resized_frame:")
        #print(resized_frame)
        datav.append(resized_frame)
    #print(file_path)
    #print(label)
    
print(len(datav))
print(len(labelsv))
x_trainv = []
y_trainv = []
x_testv = []
y_testv = []
    
for i, emotion in enumerate(emotions):
    class_start = i * 4000
    class_middle = class_start + 2000
    class_end = class_middle + 2000
    
    x_trainv.extend(datav[class_start:class_middle])
    x_testv.extend(datav[class_middle:class_end])
    
    y_trainv.extend(labelsv[class_start:class_middle])
    y_testv.extend(labelsv[class_middle:class_end])
    
del (datav)
del (labelsv)    
gc.collect()
    
    
x_trainv = np.array(x_trainv)
x_testv = np.array(x_testv)
y_trainv = np.array(y_trainv)
y_testv = np.array(y_testv)
    
print("successful data loading for video") 

x_trainv_tensor = torch.from_numpy(x_trainv).float()
x_testv_tensor = torch.from_numpy(x_testv).float()
y_trainv_tensor = torch.from_numpy(y_trainv).long()  # Assuming y_trainv contains integer class labels
y_testv_tensor = torch.from_numpy(y_testv).long()    # Assuming y_testv contains integer class labels

# Create TensorDataset
train_dataset = TensorDataset(x_trainv_tensor, y_trainv_tensor)
test_dataset = TensorDataset(x_testv_tensor, y_testv_tensor)

# Create DataLoader
batch_size = 128  # Adjust the batch size according to your needs
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Initialize the model, criterion, and optimizer
  # Modify this based on the number of emotion classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_model_path = "C:\\Users\\user.DESKTOP-HI4HHBR\\Desktop\\pretrained_model(2).pth"
vit_model = ViTEmotionClassifier(num_classes=5)  # Create an instance of your model
vit_model.load_state_dict(torch.load(pretrained_model_path))


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit_model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
vit_model.to(device)

for epoch in range(num_epochs):
    vit_model.train()
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vit_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation step
    vit_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for val_inputs, val_labels in test_dataloader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = vit_model(val_inputs)
            _, predicted = torch.max(val_outputs, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()

        accuracy = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')
