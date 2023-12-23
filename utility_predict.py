import argparse
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

def get_input_args():
    """Retrieve and parse command line arguments defined using argparse 
    module.

    1. input (str): Path to image (compulsory)
    2. checkpoint (str): Path to checkpoint (compulsory)
    3. top_k (int): Number of most likely classes
    4. category_names (str): Use mapping of categories to real names 
    5. gpu (bool): Use GPU for training
    
    Args: None

    Returns: Container with the command line arguments  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default=None, help='Path to image file')
    # Needs file name along with extension
    parser.add_argument('checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--top_k', type=int, default=3, help = 'Number of most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU for training')
    return parser.parse_args()

def check_command_line_arguments(in_args):
    """Prints all command line arguments

    Args: ArgumentParser object

    Returns: None
    """
    print("\nCommand line arguments:",
          "\n    input = ", in_args.input, 
          "\n    checkpoint = ", in_args.checkpoint, 
          "\n    top_k = ", in_args.top_k, 
          "\n    category_names = ", in_args.category_names, 
          "\n    gpu = ", in_args.gpu, 
          "\n")
    
def load_checkpoint(filepath):
    """Load the model from the given file path

    Args: File path (str)

    Returns: CNN Model Architecture
    """
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

    Args: Image path

    Returns: Numpy array of image
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])
    
    with Image.open(image) as img:
        img_tensor = preprocess(img)

    np_image = np.array(img_tensor)
    
    return np_image

def predict(image_path, model, topk):
    """Predict the class (or classes) of an image using a trained deep learning model.
    
    Args: Image path, Model, Top k classes

    Returns: Top k probabilities, top k classes
    """
    
    processed_image = process_image(image_path)
    
    image_tensor = torch.from_numpy(processed_image).unsqueeze(0).float()
    
    model.eval()
    
    with torch.no_grad():
        output = model(image_tensor)
    
    probabilities, indices = torch.topk(torch.exp(output), topk)
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[idx.item()] for idx in indices[0]]
    
    probabilities = probabilities.numpy()[0]
    classes = np.array(classes)
    
    return probabilities, classes