import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class QRCodeDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_dataset(base_path):
    first_prints = []
    second_prints = []
    
    # Load first prints
    first_print_path = os.path.join(base_path, 'Assignment Data/First Print')
    for file in os.listdir(first_print_path):
        if file.endswith('.png'):
            img_path = os.path.join(first_print_path, file)
            img = cv2.imread(img_path)
            first_prints.append(img)
    
    # Load second prints
    second_print_path = os.path.join(base_path, 'Assignment Data/Second Print')
    for file in os.listdir(second_print_path):
        if file.endswith('.jpg') or file.endswith('.png'):
            img_path = os.path.join(second_print_path, file)
            img = cv2.imread(img_path)
            second_prints.append(img)
    
    return first_prints, second_prints

def create_data_loaders(first_prints, second_prints, batch_size=32):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create labels
    y_first = np.zeros(len(first_prints))  # 0 for first prints
    y_second = np.ones(len(second_prints))  # 1 for second prints
    
    # Combine data
    X = first_prints + second_prints
    y = np.hstack((y_first, y_second))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = QRCodeDataset(X_train, y_train, transform=transform)
    test_dataset = QRCodeDataset(X_test, y_test, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, num_epochs=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels.long())
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels.long())
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_acc = running_corrects.double() / len(test_loader.dataset)
        
        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_resnet_model.pth')
        
        print()

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print("ResNet Model Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    base_path = "."
    first_prints, second_prints = load_dataset(base_path)
    
    if len(first_prints) == 0 or len(second_prints) == 0:
        print("No images were loaded. Please check if the image files exist in the correct directories.")
        return
    
    print(f"Loaded {len(first_prints)} first prints and {len(second_prints)} second prints")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(first_prints, second_prints)
    
    # Load pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: first print and second print
    model = model.to(device)
    
    # Train the model
    train_model(model, train_loader, test_loader, num_epochs=10, device=device)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_resnet_model.pth'))
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main() 