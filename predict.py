import argparse
import json
import torch
import numpy as np
from PIL import Image
from model import load_checkpoint, predict
from utils import process_image

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image using a trained model')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    model, class_to_idx = load_checkpoint(args.checkpoint_path)
    image = process_image(args.image_path)
    probs, classes = predict(image, model, args.top_k, args.gpu)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    idx_to_class = {v: k for k,v in class_to_idx.items()}
    class_names = [cat_to_name[idx_to_class[class_]] for class_ in classes]

    for i in range(len(classes)):
        print(f"Class: {class_names[i]}, Probability: {probs[i]*100:.2f}%")

if __name__ == "__main__":
    main()
