from torchvision import datasets, transforms
from torch.utils.data import DataLoader,TensorDataset
import torch
import numpy as np
import pickle
import os

img_size = 32
num_channels = 3
img_size_flat = img_size*img_size*num_channels
num_classes = 10

num_files_train = 5
num_files_validate = 1
num_images_per_file = 10000
num_images_train = num_files_train*num_images_per_file
num_images_validate = num_files_validate*num_images_per_file


def unpickle(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f,encoding='bytes')

    return data

def load_data(data):
    raw_images = data[b'data']
    raw_labels = data[b'labels']
    return raw_images,raw_labels

def convert_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])
    return images

def convert_labels(raw):
    raw_float = np.array(raw,dtype=float)
    return raw_float

def load_batch(filename):
    data = unpickle(filename)
    raw_images,raw_labels = load_data(data)
    images = convert_images(raw_images)
    labels = convert_labels(raw_labels)
    return images,labels

def read_files():
    train_images = np.zeros(shape=[num_images_train - num_images_validate, img_size, img_size, num_channels], dtype=float)
    train_labels = np.zeros(shape=[num_images_train - num_images_validate], dtype=float)
    validate_images = np.zeros(shape=[num_images_validate, img_size, img_size, num_channels], dtype=float)
    validate_labels = np.zeros(shape=[num_images_validate], dtype=float)

    t_begin = 0
    folderpath = './cifar-10-batches-py/data_batch_'
    for i in range(num_files_train-num_files_validate):
        images_batch, labels_batch = load_batch(folderpath+str(i+1))
        num_images = len(images_batch)
        t_end = t_begin+num_images

        train_images[t_begin:t_end, :] = images_batch
        train_labels[t_begin:t_end] = labels_batch
        t_begin = t_end

    v_begin = 0
    for j in range(num_files_validate):
        images_batch, labels_batch = load_batch(folderpath+str(i+1))
        num_images = len(images_batch)
        v_end = v_begin+num_images

        validate_images[v_begin:v_end, :] = images_batch
        validate_labels[v_begin:v_end] = labels_batch
        v_begin = v_end

    return train_images,train_labels, validate_images,validate_labels

def get_training_validation_loaders():
    train_images,train_labels, validate_images,validate_labels = read_files()

    tensor_train_images = torch.from_numpy(train_images).float()
    tensor_train_labels = torch.from_numpy(train_labels).float()
    tensor_validate_images = torch.from_numpy(validate_images).float()
    tensor_validate_labels = torch.from_numpy(validate_labels).float()

    train_dataset = TensorDataset(tensor_train_images,tensor_train_labels)
    validate_dataset = TensorDataset(tensor_validate_images,tensor_validate_labels)

    train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    validate_dataloader = DataLoader(validate_dataset,batch_size=32)

    return train_dataloader,validate_dataloader

def get_testing_loader():
    folderpath = './cifar-10-batches-py/test_batch'
    images,labels = load_batch(folderpath)

    tensor_test_images = torch.from_numpy(images).float()
    tensor_test_labels = torch.from_numpy(labels).float()

    test_dataset = TensorDataset(tensor_test_images,tensor_test_labels)

    test_dataloader = DataLoader(test_dataset,batch_size=32)

    return test_dataloader