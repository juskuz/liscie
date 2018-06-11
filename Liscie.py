
print('Loading libraries...')
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.color as color
import skimage.data as data
import skimage.io as io
import skimage.measure as measure
import skimage.morphology as mo
from scipy.signal import convolve2d
from skimage import data
from skimage.draw import ellipse
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
print('Libraries loaded successfully!\n')

def get_just_folders(directory):
    folders = []
    for name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, name)):
            folders.append(name)
    return folders

def get_folders_with_files(directory, folders):
    folders_with_files = []
    for folder_name in folders:
        files = list(os.listdir(directory + '/' + folder_name))
        folders_with_files.append({'name': folder_name, 'files': files})
    return folders_with_files

images_directory = sys.argv[1]
folders = get_just_folders(images_directory)
folders_with_files = get_folders_with_files(images_directory, folders)
for folder in folders_with_files:
    for image_name in folder['files']:
        image = io.imread(images_directory + '/' + folder['name'] + '/' + image_name)
        io.imshow(image)
        io.show()