import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from matplotlib.pyplot import MultipleLocator
import numpy as np


def plotfeature(path):
    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            images.append(os.path.join(root, file))
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        plt.subplot(4, 6, 1 + i)
        plt.axis('off')
        plt.imshow(img[:, :, ::-1])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


# define the function
def logs_vis(loss, acc, val_acc):
    # end = len(loss.step.values)
    # index = range(0, end, 50)
    plt.figure(figsize=(8, 4))
    plt.plot(acc.step.values, acc.value.values, label='train_acc')
    plt.plot(val_acc.step.values*281, val_acc.value.values, label='test_acc')
    plt.xlabel('Iters')
    plt.ylabel('Acc')
    plt.title('Acc curve')
    plt.yticks(np.array(range(0, 11)) / 10)
    plt.legend()
    plt.tight_layout()
    plt.savefig("acc.png")
    plt.figure(figsize=(8, 4))
    plt.plot(loss.step.values, loss.value.values, label='train_loss')
    plt.xlabel('Iters')
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.yticks(np.array(range(0, 6)))
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss.png")


if __name__ == '__main__':
    train_acc = pd.read_csv("logs/anminals_resnet50/20200317-174037train_acc.csv")
    train_loss = pd.read_csv("logs/anminals_resnet50/20200317-174037train_loss.csv")
    test_acc = pd.read_csv("logs/anminals_resnet50/20200317-174037test_acc.csv")
    # test_loss = pd.read_csv("logs/cat_vs_dog_resnet18/0.001/test_loss.csv")
    logs_vis(train_loss, train_acc, test_acc)
