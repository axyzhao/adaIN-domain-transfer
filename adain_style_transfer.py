#!/usr/bin/env python
# coding: utf-8

# ## Adaptive Instance Normalization
# 

# In[40]:


from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn.functional as F
from torch import nn, optim

from torchvision import datasets, transforms
import glob
from torchvision import utils
import datetime
import h5py


# In[42]:


data_dir = "data/PACS/"


train_files = glob.glob('data/PACS/*train.hdf5')
test_files = glob.glob('data/PACS/*test.hdf5')


filename_train = 'data/PACS/cartoon_train.hdf5'
filename_test = 'data/PACS/cartoon_test.hdf5'
filename_test = 'data/PACS/cartoon_train.hdf5'

with h5py.File(filename_train, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    # Get the data
    training_data = np.array(f['images'])
    training_labels = np.array(f['labels'])
    
with h5py.File(filename_test, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    # Get the data
    test_data = np.array(f['images'])
    test_labels = np.array(f['labels'])
    


# In[45]:


def content_tf(img):
    img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img).float()


# ### Show example image

# In[46]:


plt.imshow(transforms.ToTensor()(training_data[0]).permute(1, 2, 0))


# ### Configuration

# In[47]:


batch_size = 1
num_epochs = 10
num_output_classes = max(training_labels) + 1

## This variable determines if we normalize the layer before or after the activation functions.
normalize_after_activation = False

## Determine how often to print loss
show_every = 100

# ### Helper functions

# In[48]:


def normalize_activations(x, mean_var):
    for i in range(len(x)):
        x[i] = normalize(x[i], mean_var[i][0], mean_var[i][1])
    return torch.tensor(x)


# In[49]:


def normalize(lst, mean, std):
    a = np.sqrt(std ** 2 / np.std(lst) ** 2)
    b = mean - (np.mean(lst) * a)
    result = (lst * a) + b
    return result

# In[50]:

def get_mean_var(img):
    mean_var = []
    for i in img:
        i = i.reshape(-1)
        mean_var.append((i.mean(), i.std()))
    return mean_var


# In[51]:


def shuffle_data(training_data, training_labels):
    idx = np.arange(len(training_data))
    np.random.shuffle(idx)
    training_data_shuffled = training_data[idx]
    training_labels_shuffled = training_labels[idx]
    return training_data_shuffled, training_labels_shuffled

# ### Define model

# In[55]:


model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet34', pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_output_classes)


# ### Training

# In[56]:


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=3e-4)
criterion = torch.nn.CrossEntropyLoss()


# In[57]:


losses = []


# In[58]:


num_epochs = 10


# In[ ]:

for i in range(num_epochs):
    # for accuracy metric
    correct_count, all_count = 0, 0
    iter_ = 0
    print("\n")
    print("Starting epoch {}:".format(i))
    # shuffle data before each training epoch
    training_data, training_labels = shuffle_data(training_data, training_labels)
    for img, label in zip(training_data, training_labels):
        optimizer.zero_grad()
        # create minibatch by unsqueezing
        img = content_tf(img).unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0).long()
        assert not torch.isnan(img).any()
        # forward image through model
        probabilities = model.forward(img)
        loss_ = criterion(probabilities, label)
        losses.append(loss_)

        if iter_ % show_every == 0:
            print("Time: {}".format(datetime.datetime.now()))
            print("Loss at step {} is {}".format(iter_, loss_))
            
        # back propagate
        loss_.backward()
        optimizer.step()
        
        for i in range(len(label)):
            true_label = label.numpy()[i]
            pred_label = probabilities[i].argmax()
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
        iter_ += 1
        
    print('\n')
    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))
    torch.save(model, 'resnet34_classifier.pt')


# ### Evaluation

# In[ ]:



# for accuracy metric
correct_count, all_count = 0, 0

for img, label in zip(test_data, test_labels):
    model.eval()
    with torch.no_grad():
        # create minibatch by unsqueezing
        img = content_tf(img).unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0).long()

        # forward image through model
        probabilities = model.forward(img)
        loss_ = criterion(probabilities, label)

        if (all_count * batch_size) % show_every == 0:
            print("Time: {}".format(datetime.datetime.now()))
            print("Loss at step {} is {}".format(all_count, loss_))

        highest = probabilities.argmax(dim=1)
        for i in range(len(label)):
            true_label = label.numpy()[i]
            pred_label = probabilities[i].argmax()
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1

print('\n')
print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
#torch.save(model, 'resnet34_classifier.pt')

