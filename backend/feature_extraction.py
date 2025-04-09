import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as torchvision_models

class LSTMModel(nn.Module):
    def __init__(self, input_size=35428, hidden_size=128, num_layers=1, num_classes=2): 
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x, (torch.zeros(1, x.size(0), 128), torch.zeros(1, x.size(0), 128)))
        return self.fc(out[:, -1, :])

lstm_model = LSTMModel(input_size=35428)  
lstm_model.load_state_dict(torch.load('models/lstm_model.pth', map_location=torch.device('cpu')))
lstm_model.eval()



class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_extractor = CNNFeatureExtractor()
cnn_extractor.eval()

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None 

    image = cv2.resize(image, (128, 128))

    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(image).flatten()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform(Image.open(image_path)).unsqueeze(0)

    xception_model = torchvision_models.efficientnet_b0(weights=torchvision_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    xception_model.classifier = nn.Identity()
    xception_model.eval()
    with torch.no_grad():
        xception_features = xception_model(image_tensor).flatten().numpy()
    
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        cnn_features = cnn_extractor(image_tensor).flatten().numpy()

    return np.concatenate((hog_features, xception_features, cnn_features))
