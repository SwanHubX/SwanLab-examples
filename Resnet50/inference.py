import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F


# Function to load an image and perform necessary transforms
def process_image(image_path, image_size):
    image = Image.open(image_path).convert('RGB')
    # Define the same transforms as used during training
    preprocessing = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocessing(image).unsqueeze(0)


# Load the model with the same structure as used in training
def load_model(checkpoint_path, num_classes, device="cpu"):
    model = torchvision.models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model


# Function to predict the image class
def predict(image_path, model):
    classes = {'0': 'cat', '1': 'dog'}
    # Ensure model is in eval mode
    model.eval()
    image = process_image(image_path, 256)  # Using the image size from training
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1).squeeze()  # Apply softmax to get probabilities
    # Mapping class labels to probabilities
    class_probabilities = {classes[str(i)]: float(prob) for i, prob in enumerate(probabilities)}
    return class_probabilities


if __name__ == "__main__":
    # User-defined variables
    image_path = 'test_images/test_cat.jpg'  # replace with your image path
    checkpoint_path = 'checkpoint/best_checkpoint.pth'  # replace with your checkpoint path

    # Load the model
    num_classes = 2  # as defined in the training script
    model = load_model(checkpoint_path, num_classes)

    # Make prediction
    class_probabilities = predict(image_path, model)
    print("class_probabilities: ", class_probabilities)
