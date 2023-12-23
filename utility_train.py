import argparse
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict

def get_input_args():
    """Retrieve and parse the command line arguments defined using the 
    argparse module.
    
    1. data_dir (str): Path to the image directory being used as the dataset (compulsory)
    2. arch (str): CNN model architecture for image classification
    3. save_dir (str): Path to directory to save checkpoints 
    4. learning_rate (float): Model learning rate
    5. hidden_units (int): Units in hidden layer
    6. epochs (int): Number of epochs of the training data 
    7. gpu (bool): Use GPU for training

    Args: None

    Returns: parse_args: Container with the command line arguments     
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default = None, help='Path to image directory')
    parser.add_argument('--arch', type=str, default='vgg16', help='CNN model architecture for classifying images')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Model learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Units in hidden layer pre-classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of passes of the training data')
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU for training')
    return parser.parse_args()

def check_command_line_arguments(in_args):
    """Prints all command line arguments

    Args: ArgumentParser object

    Returns: None
    """
    print("\nCommand line arguments:",
          "\n    dir = ", in_args.data_dir, 
          "\n    arch = ", in_args.arch, 
          "\n    save_dir = ", in_args.save_dir, 
          "\n    learning_rate = ", in_args.learning_rate, 
          "\n    hidden_units = ", in_args.hidden_units, 
          "\n    epochs = ", in_args.epochs, 
          "\n    gpu = ", in_args.gpu, 
          "\n")
    
def get_data_transforms():
    """
    Defining the transformation for all the datasets in training, validation and testing sets

    Args: None

    Returns: Dictionary of transformation parameters for all datasets
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    data_transforms = {
        'train' : transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)
                                    ]),
        'valid' : transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)
                                    ]),
        'test' : transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)
                                    ])
    }
    return data_transforms

def get_image_datasets(data_transforms,train_dir, valid_dir, test_dir):
    """Loading the datasets with Image Folder for all training, validation and testing datasets
    
    Args: dictionary for data transforms, directory for all datasets
    
    Returns: Dictionary for image_datasets
    """
    image_datasets = {
        'train' : datasets.ImageFolder(train_dir, transform = data_transforms['train']),
        'valid' : datasets.ImageFolder(valid_dir, transform = data_transforms['valid']),
        'test' : datasets.ImageFolder(test_dir, transform = data_transforms['test'])
    }
    return image_datasets

    
def get_dataloaders(image_datasets):
    """Using the image datasets to define the dataloader for all datasets

    Args: Dictionary of image datasets

    Returns: Dataloaders dictionary, class_to_idx_dict
    """
    class_to_idx_dict = image_datasets['train'].class_to_idx

    batch_size = 64
    train_loader = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True)

    dataloaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

    return dataloaders, class_to_idx_dict

def get_model(arch):
    """Return model according to the input CNN architecture as string

    Args: String for model name

    Returns: CNN Model Architecture
    """
    model = getattr(models, arch)
    return model(pretrained=True)

def build_classifier(model, hidden_layer):
    """Returns classifier for the model

    Args: CNN Model Architecture, number of hidden layer units

    Returns: Classifier
    """
    in_features = model.classifier._modules['0'].in_features
    classifier = nn.Sequential(OrderedDict([
        ('dropout1', nn.Dropout(0.5)),
        ('fc1', nn.Linear(in_features, hidden_layer)),
        ('relu', nn.ReLU()),
        ('dropout2', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_layer, 102)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    return classifier

def train_and_validate_model(model,dataloaders,in_args,optimizer):
    """Train the model on training and validation datasets and print the statistics

    Args: Model, Dataloaders, Arguments, Optimizer

    Returns: None
    """
    criterion = nn.NLLLoss()

    # Move the model to the GPU according to input
    device = torch.device("cuda" if in_args.gpu else "cpu")
    model.to(device)

    epochs = in_args.epochs
    for epoch in range(epochs):
        train_loss = 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Validate the model
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        model.train()

        # Range: 0-1
        print(f"Epoch {epoch+1}/{epochs}.. "
            f"Train loss: {train_loss/len(dataloaders['train']):.3f}.. "
            f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
            f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")