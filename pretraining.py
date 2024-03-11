import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import timm
import os
import cv2
import numpy as np
from PIL import Image

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

def load_and_preprocess_data(folder_path, label_file_path, transform):
    datav = []
    labelsv = []

    if os.path.exists(folder_path) and os.path.exists(label_file_path):
        file_list = sorted(os.listdir(folder_path), key=sort_cmp)

        selected_files = file_list[::66]

        for file_name in selected_files:
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
                

        with open(label_file_path, 'r') as file:
            lines = file.readlines()[1:]

            for line in lines:
                selected_labels = [[int(digit) for digit in line.strip()[i:i+1]] for i in range(0, len(line.strip()), 66)]
                selected_labels = np.array(selected_labels, dtype=np.int64)
                labelsv.extend(selected_labels)
                
        flat_labels = [label for sublist in labelsv for label in sublist]

    
    else:
        print(str(folder_path)+" doesn't exist")

    return np.array(datav), np.array(flat_labels, dtype=np.int64)


pretrain_subject_folders = []

for i in range(10, 43):
    pretrain_subject_folders.append(f'subject{i}')

pretrain_data = []
pretrain_labels = []

for subfolder in pretrain_subject_folders:
    folder_path = f"C:\\Users\\user.DESKTOP-HI4HHBR\\Desktop\\frames(1)\\{subfolder}\\"
    label_file_path = f"C:\\Users\\user.DESKTOP-HI4HHBR\\Desktop\\labels\\{subfolder}_labels.txt"

    data, labels = load_and_preprocess_data(folder_path, label_file_path, transform)
    pretrain_data.extend(data)
    pretrain_labels.extend(labels)

print(len(pretrain_data))
print(len(pretrain_labels))

pretrain_data = np.array(pretrain_data)
pretrain_labels = np.array(pretrain_labels)


# Create TensorDataset
pretrain_dataset = TensorDataset(torch.from_numpy(pretrain_data).float(), torch.from_numpy(pretrain_labels).long())

# Create DataLoader
pretrain_batch_size = 64 
pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=pretrain_batch_size, shuffle=True)


vit_model = ViTEmotionClassifier(num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit_model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
vit_model.to(device)
print("training started:\n")
        
        
for epoch in range(num_epochs):
    print(str(epoch) + " epoch started\n")
    vit_model.train()
    total_correct = 0
    total_samples = 0
    
    for inputs, labels in pretrain_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vit_model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    # Calculate accuracy for the epoch
    accuracy = total_correct / total_samples
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

        
        
pretrained_model_path = "C:\\Users\\user.DESKTOP-HI4HHBR\\Desktop\\pretrained_model(3).pth"
torch.save(vit_model.state_dict(), pretrained_model_path)
