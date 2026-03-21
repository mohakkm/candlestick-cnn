"""
train_cnn.py
============
This script trains a simple CNN to classify candlestick patterns.

Classes:
- marubozu
- shooting_star

The CNN is built from scratch (no pretrained models) to keep things simple.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

print("Loading PyTorch (this may take 30-60 seconds on first run)...")
import torch
print("PyTorch loaded successfully!")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

# Paths
IMAGES_DIR = 'data/images'
MODEL_SAVE_PATH = 'candlestick_cnn_model.pth'

# Training parameters
BATCH_SIZE = 32          # How many images to process at once
NUM_EPOCHS = 40         # How many times to go through all data
LEARNING_RATE = 0.0003   # Reduced for longer training: lower LR with more epochs allows gradual, stable convergence

# Image settings
IMAGE_SIZE = 224
NUM_CLASSES = 2          # marubozu, shooting_star

# Class weights to handle imbalance (Hammer has fewer samples)
# Will be computed dynamically based on dataset
CLASS_WEIGHTS = None

# Early stopping to prevent overfitting
EARLY_STOPPING_PATIENCE = 8  # Stop if no improvement for 8 epochs

# Device (use GPU if available, otherwise CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# CNN MODEL DEFINITION
# ============================================================

class SimpleCNN(nn.Module):
    """
    Improved CNN architecture with better capacity to handle all patterns.
    
    Key improvements:
    - More filters in each layer for better feature extraction
    - Batch normalization to stabilize training
    - More dropout to reduce overfitting on Doji
    - Larger fully connected layer
    """
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # ===== CONVOLUTIONAL LAYERS =====
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # ===== POOLING AND ACTIVATION =====
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # ===== FULLY CONNECTED LAYERS =====
        # After 3 pooling: 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # ===== DROPOUT (increased to prevent overfitting) =====
        self.dropout1 = nn.Dropout(0.6)  # After conv layers
        self.dropout2 = nn.Dropout(0.5)  # After first FC layer
        self.dropout3 = nn.Dropout(0.3)  # After second FC layer
    
    def forward(self, x):
        """Forward pass with batch norm and increased regularization."""
        # Layer 1: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # Layer 2: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        
        # Layer 3: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout3(x)
        
        x = self.fc3(x)
        
        return x


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def get_data_transforms():
    """
    Define how to preprocess images before feeding to the CNN.
    
    Training: Add aggressive augmentation for better generalization
    Validation/Test: Just resize and normalize
    """
    # Training transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Vary brightness/contrast (market conditions)
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Slight shift
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Validation/Test transforms - no augmentation
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform


def load_datasets():
    """
    Load image datasets from folders.
    
    Expected folder structure:
    data/images/
        train/
            doji/
            hammer/
            engulfing/
        val/
            ...
        test/
            ...
    """
    train_transform, val_transform = get_data_transforms()
    
    # Load training data
    train_dataset = datasets.ImageFolder(
        root=os.path.join(IMAGES_DIR, 'train'),
        transform=train_transform
    )
    
    # Load validation data
    val_dataset = datasets.ImageFolder(
        root=os.path.join(IMAGES_DIR, 'val'),
        transform=val_transform
    )
    
    # Load test data
    test_dataset = datasets.ImageFolder(
        root=os.path.join(IMAGES_DIR, 'test'),
        transform=val_transform
    )
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset):
    """
    Create DataLoaders for efficient batch processing.
    
    DataLoader automatically:
    - Groups images into batches
    - Shuffles training data
    - Loads data in parallel (if num_workers > 0)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,        # Shuffle training data
        num_workers=0        # Use 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer):
    """
    Train the model for one complete pass through the training data.
    
    Returns: average loss and accuracy for this epoch
    """
    model.train()  # Set model to training mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        # Move data to GPU/CPU
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Clear previous gradients
        optimizer.zero_grad()
        
        # Forward pass: get predictions
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass: calculate gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate averages
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion):
    """
    Evaluate the model on validation data.
    
    Returns: average loss and accuracy
    """
    model.eval()  # Set model to evaluation mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Don't calculate gradients during validation (saves memory)
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Train the model for multiple epochs with early stopping.
    
    Prints progress and saves the best model.
    Early stopping prevents overfitting by stopping when validation accuracy
    stops improving.
    """
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("\nStarting training...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Save statistics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  EARLY STOPPING: Validation accuracy didn't improve for {EARLY_STOPPING_PATIENCE} epochs")
                print(f"  Best validation accuracy: {best_val_acc:.2f}%")
                break
        
        print("-" * 60)
    
    return train_losses, val_losses, train_accs, val_accs


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """
    Plot training and validation loss/accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    print("\nTraining history plot saved to 'training_history.png'")


def test_model(model, test_loader, class_names):
    """
    Evaluate the model on the test set and print detailed results.
    """
    model.eval()
    
    correct = 0
    total = 0
    
    # Track per-class accuracy
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    
    # Collect all predictions and labels for confusion matrix
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {100 * correct / total:.2f}%")
    print(f"Total Test Samples: {total}")
    print(f"Correctly Classified: {correct}")
    
    print("\nPer-Class Accuracy:")
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  {class_names[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    # Generate and plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("Confusion matrix saved to 'confusion_matrix.png'")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """
    Main function to train the CNN model.
    """
    print("=" * 60)
    print("CANDLESTICK PATTERN CNN TRAINING (IMPROVED)")
    print("=" * 60)
    
    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")
    
    # Step 1: Load datasets
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATASETS")
    print("=" * 60)
    
    train_dataset, val_dataset, test_dataset = load_datasets()
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # Calculate class weights to handle imbalance
    print("\n" + "=" * 60)
    print("ANALYZING CLASS DISTRIBUTION")
    print("=" * 60)
    
    class_counts = [0] * NUM_CLASSES
    for _, label in train_dataset:
        class_counts[label] += 1
    
    print("Training set class distribution:")
    for i, class_name in enumerate(train_dataset.classes):
        print(f"  {class_name}: {class_counts[i]} samples")
    
    # Step 2: Create model
    print("\n" + "=" * 60)
    print("STEP 2: CREATING MODEL")
    print("=" * 60)
    
    model = SimpleCNN(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    
    # Print model summary
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Step 3: Define loss function and optimizer
    print("\n" + "=" * 60)
    print("STEP 3: SETTING UP TRAINING")
    print("=" * 60)
    
    # Standard CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Loss function: CrossEntropyLoss")
    print("Optimizer: Adam")
    print("  ✓ Aggressive data augmentation for robustness")
    print("  ✓ Batch normalization for stable training")
    print("  ✓ Increased dropout to prevent Doji overfitting")
    print("  ✓ Early stopping to prevent overfitting")
    
    # Step 4: Train the model
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING")
    print("=" * 60)
    
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )
    
    # Step 5: Plot training history
    print("\n" + "=" * 60)
    print("STEP 5: VISUALIZING RESULTS")
    print("=" * 60)
    
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Step 6: Test the model
    print("\n" + "=" * 60)
    print("STEP 6: TESTING")
    print("=" * 60)
    
    # Load best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_model(model, test_loader, train_dataset.classes)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print("You can now use this model in backtest.py")


if __name__ == "__main__":
    main()
