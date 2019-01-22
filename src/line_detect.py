# In[1]
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import scipy
import copy
from src.logistic_regression import predict
from src.lr_utils import load_dataset
from PIL import Image
from scipy import ndimage, io

# In[2]d
image = ndimage.imread("images/paper_1_cropped.jpg", flatten=True)
print(image)
image = 255 - image
plt.imshow(image, cmap="Greys")
print(np.shape(image))
print(image)
scipy.io.savemat('datasets/image.mat', mdict={'image': image})


# In[3]
first = image[:, 0]
last = image[:, len(image[1]) - 1]
first_col = copy.copy(first)
last_col = copy.copy(last)

line_width = len(image) * 0.002
div_count = 5
div_width = int(len(first_col)/div_count)
remainder = len(first_col) % div_count

for c in range(2):
    if c == 0:
        col = first_col
    else:
        col = last_col
    sections = []
    for i in range(div_count):
        if i == div_count - 1:
            section = col[i*div_width:(i+1)*div_width + remainder]
        else:
            section = col[i*div_width:(i+1)*div_width]
        print(np.shape(section))
        imax = np.amax(section)
        imin = np.amin(section)
        irange = imax - imin
        section -= imin
        section *= 255/irange
        hist = plt.hist(section, bins=10)
        diffs = []
        for j in range(len(hist[0])-2):
            first = hist[0][j]
            second = hist[0][j+1]
            diffs.append(first/second)
        diff_max = np.amax(diffs)
        # print(diffs)
        diff_max_index = diffs.index(diff_max) + 1
        print(hist)
        print(hist[1][diff_max_index])
        for i in range(len(section)):
            if section[i] < hist[1][diff_max_index]:
                section[i] = 0
            else:
                section[i] = 255
        print(np.shape(section))
        sections = np.append(sections, section)
        print(np.shape(sections))
    starts = np.where(np.diff(sections) == 255)
    starts = starts[0]
    ends = np.where(np.diff(sections) == -255)
    ends = ends[0]
    if sections[0] == 255:
        starts = np.insert(starts, 0, 0)
    if sections[len(sections) - 1] == 255:
        ends = np.append(ends, len(sections) - 1)
    widths = starts - ends
    print(sections[len(sections) - 1])
    print(starts)
    print(ends)
    # print(widths)
    if c == 0:
        scipy.io.savemat('datasets/first_col.mat', mdict={'col': sections})
    else:
        scipy.io.savemat('datasets/last_col.mat', mdict={'col': sections})
