import torch
import torchvision
from torchvision.models import ResNet50_Weights
import swanlab
from torch.utils.data import DataLoader
from load_datasets import DatasetLoader
import os


# Define train function
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for iter, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, iter + 1, len(TrainDataLoader),
                                                                      loss.item()))
        swanlab.log({"train_loss": loss.item()})


# Define test function
def test(model, device, test_loader, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    print('Accuracy: {:.2f}%'.format(accuracy))
    swanlab.log({"validation_acc": accuracy}, step=epoch)
    return accuracy


if __name__ == "__main__":
    num_epochs = 20
    lr = 1e-4
    batch_size = 16
    image_size = 256
    num_classes = 2
    seed = 2024

    try:
        use_mps = torch.backends.mps.is_available()
    except AttributeError:
        use_mps = False

    if torch.cuda.is_available():
        device = "cuda"
    elif use_mps:
        device = "mps"
    else:
        device = "cpu"

    torch.manual_seed(seed)

    # Initialize swanlab
    swanlab.init(
        experiment_name="ResNet50",
        description="Train ResNet50 for cat and dog classification.",
        config={
            "datasets": "cats_and_dogs",
            "model": "resnet50",
            "optim": "Adam",
            "criterion": "CrossEntropyLoss",
            "lr": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "image_size": image_size,
            "num_class": num_classes,
            "device": device,
            "seed": seed,
            "augmentation": "RandomHorizontalFlip+RandomRotation(15)+ColorJitter"
        }
    )

    TrainDataset = DatasetLoader("datasets/train.csv", image_size=(image_size, image_size), mode="train")
    ValDataset = DatasetLoader("datasets/val.csv", image_size=(image_size, image_size), mode="test")
    TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    ValDataLoader = DataLoader(ValDataset, batch_size=1, shuffle=False)

    # Load the pre-trained ResNet50 model
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Replace the last fully connected layer.
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    # Train
    model.to(torch.device(device))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_accuracy = 0

    for epoch in range(1, num_epochs + 1):
        train(model, device, TrainDataLoader, optimizer, criterion, epoch)  # Train for one epoch

        if epoch % 4 == 0:  # Test every 4 epochs
            accuracy = test(model, device, ValDataLoader, epoch)

            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if not os.path.exists("checkpoint"):
                    os.makedirs("checkpoint")
                torch.save(model.state_dict(), 'checkpoint/best_checkpoint.pth')
                print("Saved better model with accuracy: {:.2f}%".format(best_accuracy))

    print("Training complete")
