import os
import torch
from torch.utils.data import DataLoader
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score


def log_metrics(epoch, loss_train, loss_val, train_accuracy, val_accuracy, filename):
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Val Loss", "Train Accuracy", "Val Accuracy"])

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss_train, loss_val, train_accuracy, val_accuracy])

def save_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, train_folder):
    plt.figure(figsize=(12, 5))

    # Courbe de perte
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Évolution de la Loss")
    plt.legend()

    # Courbe d'accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Évolution de l'Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(train_folder + "learning_curves.png") 
    plt.close()
    print(f"Courbes d'apprentissage enregistrées")


def train_model(model, train_loader, val_loader, test_loader, num_epochs, device, optimizer, criterion, scheduler=None):
    
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Find the next available training folder number
    i = 1
    while os.path.exists(f"training_{i}"):
        i += 1
    train_folder = f"training_{i}/"
    os.makedirs(train_folder)
    print(f"Created new training folder: {train_folder}")
    best_val_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_batches
        train_acc = correct / total
        val_acc = val_correct / val_total

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {running_loss/len(train_loader):.3f}')
        print(f'Training Accuracy: {train_acc:.2f}')
        print(f'Validation Loss: {val_loss:.3f}')
        print(f'Validation Accuracy: {val_acc:.2f}')

        log_metrics(epoch, running_loss/len(train_loader) ,val_loss, train_acc, val_acc, f"train_metrics.csv")

        train_losses.append(running_loss/len(train_loader))
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{train_folder}model.pth")

        if scheduler is not None:
            scheduler.step()
        
    save_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, train_folder)

    # Test phase
    model.load_state_dict(torch.load(f"{train_folder}model.pth"))
    model.eval()
    test_correct = 0
    test_total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_acc = test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}')

    # Plot confusion matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{train_folder}confusion_matrix.png")
    plt.close()
