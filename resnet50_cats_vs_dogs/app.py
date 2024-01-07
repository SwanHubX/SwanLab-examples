import gradio as gr
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision


# Load the model with the same structure as used in training
def load_model(checkpoint_path, num_classes):
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

    model = torchvision.models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model

# Function to load an image and perform necessary transforms
def process_image(image, image_size):
    # Define the same transforms as used during training
    preprocessing = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocessing(image).unsqueeze(0)
    return image


# Function to predict the image class and return probabilities


def predict(image):
    classes = {'0': 'cat', '1': 'dog'}  # Update or extend this dictionary based on your actual classes
    image = process_image(image, 256)  # Using the image size from training
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1).squeeze()  # Apply softmax to get probabilities
    # Mapping class labels to probabilities
    class_probabilities = {classes[str(i)]: float(prob) for i, prob in enumerate(probabilities)}
    return class_probabilities


# Define the path to your model checkpoint
checkpoint_path = 'checkpoint/best_checkpoint.pth'  # replace with your checkpoint path
num_classes = 2  # as defined in the training script
model = load_model(checkpoint_path, num_classes)

# Define the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=num_classes),
    title="Cat vs Dog Classifier",
    examples=["test_images/test_cat.jpg", "test_images/test_dog.jpg"]
)

if __name__ == "__main__":
    iface.launch()
