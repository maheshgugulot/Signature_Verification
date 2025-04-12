import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

cnn_extractor = CNNFeatureExtractor()
cnn_extractor.eval()

class LSTMModel(nn.Module):
    def __init__(self, input_size=34148, hidden_size=128, num_layers=1, num_classes=2): 
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 128)
        c0 = torch.zeros(1, x.size(0), 128)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


augmentation = transforms.Compose([
    transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), fill=255),
    transforms.Resize((128, 128))
])

def extract_features(image_path):
    image = Image.open(image_path).convert('L')
    image_aug = augmentation(image)
    image_np = np.array(image_aug)

    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(image_np).flatten()

    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        cnn_features = cnn_extractor(image_tensor).flatten().numpy()

    return np.concatenate((hog_features, cnn_features))

def predict(image_path):
    features = extract_features(image_path)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(1)  
    with torch.no_grad():
        output = lstm_model(features_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class
