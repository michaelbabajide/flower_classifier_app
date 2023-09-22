import argparse
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from model import build_model
from utils import load_data

def train_model(data_dir, save_dir, architecture, learning_rate, hidden_units, epochs, gpu):
    # Load and preprocess data
    dataloaders, class_to_idx = load_data(data_dir)

    # Build the model
    model = build_model(architecture, hidden_units, class_to_idx)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Train the model
    model = model.to('cuda' if gpu else 'cpu')
    model, optimizer = train(model, criterion, optimizer, dataloaders['train'], dataloaders['valid'], epochs, gpu)

    # Save the model checkpoint
    checkpoint = {
        'architecture': architecture,
        'class_to_idx': class_to_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_dir)

def train(model, criterion, optimizer, train_loader, valid_loader, epochs, gpu):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cuda' if gpu else 'cpu'), labels.to('cuda' if gpu else 'cpu')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        valid_loss, accuracy = validate(model, criterion, valid_loader, gpu)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {running_loss/len(train_loader):.4f} - Validation Loss: {valid_loss:.4f} - Validation Accuracy: {accuracy*100:.2f}%")

    return model, optimizer

def validate(model, criterion, valid_loader, gpu):
    model.eval()
    valid_loss = 0.0
    accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to('cuda' if gpu else 'cpu'), labels.to('cuda' if gpu else 'cpu')
            outputs = model(inputs)
            valid_loss += criterion(outputs, labels).item()

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    return valid_loss/len(valid_loader), accuracy/len(valid_loader)

def main():
    parser = argparse.ArgumentParser(description='Train a deep learning model on a dataset')
    parser.add_argument('data_dir', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
    parser.add_argument('--architecture', type=str, default='vgg16', help='Model architectureitecture (e.g., vgg16)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()

    train_model(args.data_dir, args.save_dir, args.architecture, args.learning_rate, args.hidden_units, args.epochs, args.gpu)

if __name__ == "__main__":
    main()
