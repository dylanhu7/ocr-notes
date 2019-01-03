# In[1]
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import scipy
from src.logistic_regression import predict
from src.lr_utils import load_dataset
from PIL import Image
from scipy import ndimage, io

# In[2]
image = np.array(ndimage.imread("images\\a_2.jpg", flatten=True))
image = 255 - image
image = scipy.misc.imresize(image, size=(28, 28))
fl_image = np.asfarray(image)
imax = np.amax(image)
imin = np.amin(image)
irange = imax - imin
fl_image -= imin
fl_image *= 255/irange
image = fl_image.astype(int)
image = image.reshape(784, 1)
# Should find matrix operation that does this more efficiently
for i in range(784):
    if(image[i] < 20):
        image[i] = 0
    elif(image[i] >= 20):
        image[i] = 255
image = np.reshape(image, (28, 28))

plt.imshow(image, cmap="Greys")
print(image)

# In[3]
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
index = 126582
image = train_set_x_orig[index]
plt.imshow(train_set_x_orig[index], cmap="Greys")
params = scipy.io.loadmat("parameters\params.mat")
w = params["a"]["params"][0, 0]["w"][0, 0]
b = params["a"]["params"][0, 0]["b"][0, 0]
image = image.reshape(784, 1)
image = image/255
print(np.shape(image))
y, a = predict(w, b, image)
print(y)
print(a)
