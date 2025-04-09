import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import torch

resnet = models.resnet50(pretrained=True)
resnet.eval()
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]    
    )
])


def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(img_tensor).squeeze().numpy()
    return features.flatten()


def calculate_similarity(original_path, forged_path):
    emb1 = get_embedding(original_path)
    emb2 = get_embedding(forged_path)
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    print(f"Similarity score: {similarity}")
    return similarity


def show_differences(genuine_img_path, forged_img_path):
    genuine_img = cv2.imread(genuine_img_path, 0)
    forged_img = cv2.imread(forged_img_path, 0)

    if genuine_img is None or forged_img is None:
        print(f"Failed to load images:\nGenuine: {genuine_img_path}\nForged: {forged_img_path}")
        return None

    # Resize images to larger dimensions
    new_size = (512, 512)
    genuine_resized = cv2.resize(genuine_img, new_size)
    forged_resized = cv2.resize(forged_img, new_size)

    difference = cv2.absdiff(genuine_resized, forged_resized)
    _, thresh_diff = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    forged_image_color = cv2.cvtColor(forged_resized, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            # Make radius smaller (e.g., scale down or set max)
            radius = min(int(radius * 0.4), 10)
            cv2.circle(forged_image_color, center, radius, (0, 0, 255), 2)

    rgb_image = cv2.cvtColor(forged_image_color, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    return pil_img



def visualize_difference(image_path):
    image_path = os.path.normpath(image_path)
    filename = os.path.basename(image_path)

    if "forgeries" in filename:
        original_filename = filename.replace("forgeries", "original")
        original_path = os.path.join(
            "C:/Users/mahes/Desktop/Final_Project/Cedar/full_org", 
            original_filename
        )
        return show_differences(original_path, image_path)

    else:
        print("Filename must contain'forgeries'.")
        return None
def relevance_score(image_path):
    filename = os.path.basename(image_path)
    if "original" in filename:
        group_number = filename.split('_')[1]
        reference_path = f"C:/Users/mahes/Desktop/Final_Project/Cedar/full_org/original_{group_number}_1.png"
        similarity = calculate_similarity(reference_path, image_path)
        print(f"Similarity between {image_path} and {reference_path}: {similarity}")
        return similarity
    else:
        print("Filename must contain 'original'.")
        return None