from time import time
import torch
from utility_train import *
import os

def main():
    start_time = time()
    in_args = get_input_args()

    # Sets the parameter according to the input and availablity of GPU on user's system
    in_args.gpu = in_args.gpu and torch.cuda.is_available()
    check_command_line_arguments(in_args)

    # Creating a directory if it does not exist
    if in_args.save_dir is not None:
        if not os.path.exists(in_args.save_dir):
            os.makedirs(in_args.save_dir)

    data_dir = in_args.data_dir
    train_dir = os.path.join(data_dir,'train')
    valid_dir = os.path.join(data_dir,'valid')
    test_dir = os.path.join(data_dir,'test')

    data_transforms = get_data_transforms()

    image_datasets = get_image_datasets(data_transforms,train_dir, valid_dir, test_dir)

    dataloaders, class_to_idx = get_dataloaders(image_datasets)

    model = get_model(in_args.arch)
    model.class_to_idx = class_to_idx
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = build_classifier(model, in_args.hidden_units)

    optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)
    train_and_validate_model(model, dataloaders, in_args, optimizer)

    # Saving the model
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'model' : model,
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': in_args.epochs
    }
    
    #File Name: checkpoint.pth
    if in_args.save_dir is None:
        torch.save(checkpoint, 'checkpoint.pth')
    else:
        torch.save(checkpoint, os.path.join(in_args.save_dir,'checkpoint.pth') )

    end_time = time()
    total_time = end_time - start_time
    print("Total Elapsed Runtime: {} seconds".format(total_time))

if __name__ == "__main__":
    main()