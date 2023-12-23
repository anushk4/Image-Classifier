from time import time
from utility_predict import *
import json

def main():
    start_time = time()

    in_args = get_input_args()

    # Sets the parameter according to the input and availablity of GPU on user's system
    in_args.gpu = in_args.gpu and torch.cuda.is_available()
    check_command_line_arguments(in_args)

    category_names = in_args.category_names

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    pathname = in_args.input

    model = load_checkpoint(in_args.checkpoint)

    probs, classes = predict(pathname, model, topk=in_args.top_k)
    class_names = [cat_to_name[class_] for class_ in classes]

    print('Filepath to image: ', pathname,
          '\nTop k classes: ', class_names,
          '\nTop k probabilities (0 - 10): ', probs,
          '\nFlower name: ', class_names[0],
          '\nClass Probability for the flower (0 - 1): ', probs[0])   
    
    end_time = time()
    total_time = end_time - start_time
    print("\nTotal Elapsed Runtime: {} seconds".format(total_time))

if __name__ == "__main__":
    main()