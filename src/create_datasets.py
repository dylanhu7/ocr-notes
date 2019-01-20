# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io
import os

# Creates an h5 file for letters called letter_train.h5 in the datasets folder by restructuring the EMNIST letter dataset, which is in Matlab format
# Loads Matlab file
letter_mat = scipy.io.loadmat("datasets/emnist-letters.mat")

# Un-comment to see an example of an image
# image1 = np.reshape(letter_mat["dataset"]["train"][0,0]["images"][0,0][0], (28,28))
# plt.imshow(image1.T, cmap = "Greys")

# Creates train dataset
# Creates new array and reshapes each row of the original dataset into 28x28 matrices, each representing an image
letter_image_data = []
letter_image_labels = []
for i in range(len(letter_mat["dataset"]["train"][0, 0]["labels"][0, 0])):
    value = letter_mat["dataset"]["train"][0, 0]["labels"][0, 0][i][0]
    image = np.reshape(
        letter_mat["dataset"]["train"][0, 0]["images"][0, 0][i], (28, 28)).T
    if value == 1:
        for i in range(25):
            letter_image_labels.append(1)
            letter_image_data.append(image)
    else:
        letter_image_labels.append(0)
        letter_image_data.append(image)
randomize = np.arange(len(letter_image_data))
np.random.shuffle(randomize)
for i in range(len(letter_image_data)):
    letter_image_data[i] = letter_image_data[randomize[i]]
    letter_image_labels[i] = letter_image_labels[randomize[i]]
# # Writes a new h5 file from the new array for compatibility with the NN
os.remove("datasets/letter_train.h5")
letter_train_set = h5py.File("datasets/letter_train.h5", 'w')
letter_train_set.create_dataset("train_set_x", data=letter_image_data)
letter_train_set.create_dataset("train_set_y", data=letter_image_labels)
letter_train_set.create_dataset("list_classes", data=[0, 1])
letter_train_set.close()


# Creates test dataset
# Creates new array and reshapes each row of the original dataset into 28x28 matrices, each representing an image
letter_image_data = []
for i in range(len(letter_mat["dataset"]["test"][0, 0]["images"][0, 0])):
    letter_image_data.append(np.reshape(
        letter_mat["dataset"]["test"][0, 0]["images"][0, 0][i], (28, 28)).T)
# Creates new array for image labels (0-25 for letters)
letter_image_labels = []
for i in range(len(letter_mat["dataset"]["test"][0, 0]["labels"][0, 0])):
    value = letter_mat["dataset"]["test"][0, 0]["labels"][0, 0][i][0]
    if value == 1:
        letter_image_labels.append(1)
    else:
        letter_image_labels.append(0)

# Writes a new h5 file from the new array for compatibility with the NN
os.remove("datasets/letter_test.h5")
letter_test_set = h5py.File("datasets/letter_test.h5", 'w')
letter_test_set.create_dataset("test_set_x", data=letter_image_data)
letter_test_set.create_dataset("test_set_y", data=letter_image_labels)
letter_test_set.create_dataset("list_classes", data=[0, 1])
letter_test_set.close()
