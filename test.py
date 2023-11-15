import numpy as np
import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchviz import make_dot

from model import encoder_front, classifier
import model


def evaluator(y_test, y_pred):
    num_digits = y_test.shape[1]

    #    Get the labels from y_test
    y_labels = np.argmax(y_test, axis=1)

    #    Make the confusion matrix
    confusion_matrix = np.zeros((num_digits, num_digits), dtype=int)
    for true, pred in zip(y_labels, y_pred):
        confusion_matrix[true, pred] += 1

    #    Calculate accuracy
    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

    #    Calculate precision
    precision = np.zeros(num_digits)
    for i in range(num_digits):
        true_positives = confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - true_positives
        precision[i] = true_positives / (true_positives + false_positives)

    #    Calculate recall
    recall = np.zeros(num_digits)
    for i in range(num_digits):
        true_positives = confusion_matrix[i, i]
        false_negatives = confusion_matrix[i, :].sum() - true_positives
        recall[i] = true_positives / (true_positives + false_negatives)

    #    Calculate F1
    f1 = np.zeros(num_digits)
    for i in range(num_digits):
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    #    Print
    print(confusion_matrix)
    print("Accuracy: ", round(accuracy, 3))
    print("Precision (digit-wise): ", precision)
    print("Precision (mean): ", np.mean(precision))
    print("Recall (digit-wise): ", recall)
    print("Recall (mean): ", np.mean(recall))
    print("F1 (digit-wise): ", f1)
    print("F1 (mean): ", np.mean(f1))


####################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--l', type=str, default='encoder.pth')
parser.add_argument('--s', type=str, default='classifier.pth')
parser.add_argument('--d', type=str, default='cpu')

args = parser.parse_args()

device = torch.device(args.d)

transform = transforms.Compose([
    transforms.Resize((69, 69)),
    transforms.RandomCrop((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)

# test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)



encoder = model.encoder_front.encoder
encoder.load_state_dict(torch.load(args.l, map_location=args.d))
classify = model.encoder_front.front
classify.load_state_dict(torch.load(args.s, map_location=args.d))
model = model.classifier(encoder, classify)

model.to(device=device)
model.eval()

print("Model loaded succesfully")
#class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_names = ["beaver", "dolphin", "otter", "seal", "whale", "aquarium fish", "flatfish", "ray", "shark", "trout", "orchids", "poppies", "roses", "sunflowers", "tulips", "bottles", "bowls", "cans", "cups", "plates", "apples", "mushrooms", "oranges", "pears", "sweet peppers", "clock", "computer keyboard", "lamp", "telephone", "television", "bed", "chair", "couch", "table", "wardrobe", "bee", "beetle", "butterfly", "caterpillar", "cockroach", "bear", "leopard", "lion", "tiger", "wolf", "bridge", "castle", "house", "road", "skyscraper", "cloud", "forest", "mountain", "plain", "sea", "camel", "cattle", "chimpanzee", "elephant", "kangaroo", "fox", "porcupine", "possum", "raccoon", "skunk", "crab", "lobster", "snail", "spider", "worm", "baby", "boy", "girl", "man", "woman", "crocodile", "dinosaur", "lizard", "snake", "turtle", "hamster", "mouse", "rabbit", "shrew", "squirrel", "maple", "oak", "palm", "pine", "willow", "bicycle", "bus", "motorcycle", "pickup truck", "train", "lawn-mower", "rocket", "streetcar", "tank", "tractor"]

with torch.no_grad():
    count = 0
    top_1_count = 0
    top_5_count = 0
    for i, (img, label) in enumerate(test_loader):
        image_to_display = img[0].permute(1, 2, 0)
        img = img.to(device)
        label = label.float().to(device)

        output = model(img)
        if (output.argmax().item() == int(label.item())):
            count += 1
        else:
            top_1_count += 1

            _, top5_classes = output.topk(5, largest=True)
            if int(label.item()) in top5_classes:
                top_5_count += 1


    print("The acccuracy of the model is: ", 100 * (count / 10000))
    print("Top 1 Error: ", 100 * (top_1_count / 10000))
    print("Top 5 Error: ", 100 * (top_5_count / 10000))

# evaluator(labels, img_prediction)

input_tensor = torch.randn(1, 3, 64, 64).to(device)  # Adjust the shape based on your model's input size

# # Create an SVG representation of the model
dot = make_dot(model(input_tensor), params=dict(model.named_parameters()))

# # Save the SVG file
dot.format = 'svg'
dot.render('encoder_model_architecture')
