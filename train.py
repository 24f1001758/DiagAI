
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from PIL import Image

# Preparation of Dataset 
class XRayDataset(Dataset):
    def __init__(self, root_dir, mode="train", transform=None):
        self.root_dir = os.path.join(root_dir, mode)
        self.image_files = []
        self.labels = []
        self.transform = transform
        
        # Load images and labels
        for label, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(class_dir, img_file))
                    self.labels.append(label)  # 0 = NORMAL, 1 = PNEUMONIA

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]
        
        # To load image
        image = Image.open(img_path).convert("RGB")  # Convert grayscale to 3-channel RGB
        
        if self.transform:
            image = self.transform(image)

        return image, label

# Defining model
class DiagnosticCNN(nn.Module):
    def __init__(self):
        super(DiagnosticCNN, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # 2 classes: Normal, Pneumonia

    def forward(self, x):
        return self.resnet(x)

# GPU help
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading model
model = DiagnosticCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Loading Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = XRayDataset(root_dir="chest_xray", mode="train", transform=transform)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training the model
if not os.path.exists("model.pth"):
    print("Training Model...")
    model.train()
    
    for epoch in range(5):  
        running_loss = 0.0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(data_loader)}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved!")

else:
    print("Loading saved model...")
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

# Predicting image
def predict_xray(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  

    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output).item()

    return "Normal X-ray" if predicted_class == 0 else "Pneumonia Detected"

# Test Prediction 
test_image = "chest_xray/test/PNEUMONIA/person1_virus_7.jpeg"  
print("Prediction:", predict_xray(test_image))
