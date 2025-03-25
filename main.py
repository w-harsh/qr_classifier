import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

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

def visualize_samples(first_prints, second_prints, num_samples=3):
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    for i in range(num_samples):
        axes[0, i].imshow(cv2.cvtColor(first_prints[i], cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"First Print {i+1}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(cv2.cvtColor(second_prints[i], cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f"Second Print {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def extract_features(image):
    features = []
    
    # Basic image statistics
    # Mean and standard deviation of pixel values
    mean_values = np.mean(image, axis=(0, 1))
    std_values = np.std(image, axis=(0, 1))
    features.extend(mean_values)
    features.extend(std_values)
    
    # Image gradient features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    features.append(np.mean(gradient_magnitude))
    features.append(np.std(gradient_magnitude))
    
    # Noise estimation features (might capture reprinting artifacts)
    # Using Laplacian filter to detect edges/noise
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(np.mean(np.abs(laplacian)))
    features.append(np.std(laplacian))
    
    # Texture features using GLCM (Gray-Level Co-occurrence Matrix)
    # This would require additional implementation
    
    # Print quality features - analyzing line sharpness
    # This would require isolating QR code edges and measuring spread
    
    return np.array(features)

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(f"{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\n")

# Load and visualize the dataset
if __name__ == "__main__":
    base_path = "."  # Current directory
    first_prints, second_prints = load_dataset(base_path)
    
    # Check if we have any images loaded
    if len(first_prints) == 0 or len(second_prints) == 0:
        print("No images were loaded. Please check if the image files exist in the correct directories.")
    else:
        print(f"Loaded {len(first_prints)} first prints and {len(second_prints)} second prints")
        visualize_samples(first_prints, second_prints)
        
        # Extract features for all images
        print("Extracting features from images...")
        X_first = np.array([extract_features(img) for img in first_prints])
        X_second = np.array([extract_features(img) for img in second_prints])
        
        # Create labels
        y_first = np.zeros(len(X_first))  # 0 for first prints
        y_second = np.ones(len(X_second))  # 1 for second prints
        
        # Combine data
        X = np.vstack((X_first, X_second))
        y = np.hstack((y_first, y_second))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        print("Training Random Forest model...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Train SVM
        print("Training SVM model...")
        svm_model = SVC(kernel='rbf', probability=True)
        svm_model.fit(X_train, y_train)
        
        # Evaluate models
        print("Evaluating models...")
        rf_preds = rf_model.predict(X_test)
        svm_preds = svm_model.predict(X_test)
        
        evaluate_model(y_test, rf_preds, "Random Forest")
        evaluate_model(y_test, svm_preds, "SVM")
