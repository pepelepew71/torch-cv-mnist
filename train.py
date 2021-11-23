"""
"""

import argparse
import os
import pathlib

import imgaug
from imgaug import augmenters as iaa
import numpy as np
import torch
from torchvision import transforms

from dataset import DataSetMnist
import model

def worker_init_fn(worker_id):
    imgaug.seed(np.random.get_state()[1][0] + worker_id)

def get_acc_test(net, dataloader, device):
    """
    Get accuracy of test data
    """
    net.eval()
    correct = 0
    with torch.no_grad():
        for xs, ys in dataloader:
            xs, ys = xs.to(device), ys.to(device)
            out_labels = net(xs)  # <<< two output from DANN
            pred = out_labels.max(1, keepdim=True)[1]
            correct += pred.eq(ys.view_as(pred)).sum().item()
    acc = correct / len(dataloader.dataset)
    return acc

PWD = str(pathlib.Path(__file__).parent.absolute())

if __name__ == "__main__":

    # -- args
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_train', type=str, default="", help="")
    parser.add_argument('--folder_test', type=str, default="", help="")
    parser.add_argument('--batch_size', type=int, default=64, help="")
    parser.add_argument('--num_workers', type=int, default=6, help="")
    parser.add_argument('--num_epochs', type=int, default=10, help="")
    args = parser.parse_args()

    args.folder_train = "/home/ych/tmp/mnist/train"
    args.folder_test = "/home/ych/tmp/mnist/test"

    assert os.path.exists(args.folder_train), print(f'{args.folder_train} is not exists')
    assert os.path.exists(args.folder_test), print(f'{args.folder_test} is not exists')

    # -- datasets, dataloader
    tf_train = transforms.Compose([
        iaa.Affine(translate_percent={"x": (-0.3, 0.3)}).augment_image,
        iaa.MultiplyBrightness(mul=(0.65, 1.35)).augment_image,
        iaa.AddToHueAndSaturation(value_hue=(-50, 50), value_saturation=(-255, 255)).augment_image,
        iaa.GaussianBlur(sigma=(0.0, 0.1)).augment_image,
        iaa.AdditiveGaussianNoise(scale=0.1*255).augment_image,
        # transforms.ToPILImage(),
        # transforms.ColorJitter(),
        # transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.5, scale=(0.05, 0.06), ratio=(0.1, 0.11)),
        transforms.RandomErasing(p=0.5, scale=(0.05, 0.06), ratio=(5.0, 5.01)),
    ])

    tf_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.RandomErasing(p=0.5, scale=(0.05, 0.06), ratio=(0.1, 0.11)),
        # transforms.RandomErasing(p=0.5, scale=(0.05, 0.06), ratio=(5.0, 5.01)),
    ])

    dataset_train = DataSetMnist(folder=args.folder_train, transform=tf_train)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        worker_init_fn=worker_init_fn, pin_memory=True)

    dataset_test = DataSetMnist(folder=args.folder_test, transform=tf_test)
    dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # -- device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- model
    net = model.SimpleNet4().to(device=device)

    # -- Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # -- optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # -- Start training
    print("Starting Training Loop...")

    for epoch in range(0, args.num_epochs+1):
        net.train()
        loss_epoch = list()
        for i, data in enumerate(dataloader_train):
            net.zero_grad()
            imgs, labels = data[0].to(device=device), data[1].to(device=device)
            out_label = net(imgs)
            loss = criterion(out_label, labels)
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())

        # -- logging
        loss_epoch_mean = np.mean(loss_epoch)
        acc_test = get_acc_test(net=net, dataloader=dataloader_test, device=device)
        msg = (
            f'epoch: {epoch}, '
            f'Loss: {loss_epoch_mean:.4f}, '
            f'Acc Test: {acc_test:.4f}, '
        )
        print(msg)

    torch.save(net.state_dict(), f=os.path.join(PWD, "net.pth"))
