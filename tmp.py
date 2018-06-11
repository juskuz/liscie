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


########## liczenie kątów/narożników ##########
def count_corners(img, preview = False):
	coords = corner_peaks(corner_harris(image_grey), min_distance=3)# corner_peaks narożniki w obrazie odpowiedzi o miary rogu
	coords_subpix = corner_subpix(image_grey, coords, window_size=13)
	if preview:
		plt.show()
		fig, ax = plt.subplots()
		ax.imshow(image_grey, interpolation='nearest', cmap=plt.cm.gray)
		ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=10) # markersize - niebieskie kółka
		ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15) #markersize - wielkosc znacznika czerwone iksy
	return len(coords_subpix) #liczy kąty wklęsłe i wypukłe

########## linie w szkielecie - średnio sprawdzająca się cecha##########
def count_lines_in_skeleton(image, preview = False):
	skeleton = mo.skeletonize(image < 1)
	if preview:
		io.imshow(skeleton)
		io.show()
	lines = [item for sublist in skeleton for item in sublist]
	return lines.count(True)

def find_biggest_region(image):
    mask = (image < 0.6) #0.7
    labels = measure.label(mask)
    areas = []
    for i, e in enumerate(skimage.measure.regionprops(labels)):
        areas.append(e.area)
    if len(areas) == 0:
        return 0
    return max(areas)

def preprocess_image(image):
    image_cutted = image[:550, :550]  # przycięcie zdjęć, może 570
    image_grey = color.rgb2gray(image_cutted)  # zamiana na szary
    return image_grey

def get_just_folders(directory):
    folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return folders

def get_folders_with_files(directory, folders):
    folders_with_files = []
    for folder_name in folders:
        files = list(os.listdir(directory + '/' + folder_name))
        folders_with_files.append({'name': folder_name, 'files': files})
    return folders_with_files

images_directory = sys.argv[1] # plik wejściowy - biblioteka obrazów
output_file = sys.argv[2] # plik wyjściowy cechami wszystkich zdjęć we wszystkich folderach
if(len(sys.argv) > 3):
    output_file2 = sys.argv[3] # plik wyjściowy ze średnimi z folderów(gatunków)





''' http://scikit-image.org/docs/dev/auto_examples/edges/plot_convex_hull.html#sphx-glr-auto-examples-edges-plot-convex-hull-py
    image = invert(data.horse())
    chull = convex_hull_image(image)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()
    ax[0].set_title('Original picture')
    ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_axis_off()
    ax[1].set_title('Transformed picture')
    ax[1].imshow(chull, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].set_axis_off()
    plt.tight_layout()
    plt.show()
    return 
'''

output_for_files = []
output_for_categories = []

folders = get_just_folders(images_directory)
folders_with_files = get_folders_with_files(images_directory, folders)
for folder in folders_with_files:
    regions = []
    corners_counts = []
    lines_counts = []
    for image_name in folder['files']:
        image = io.imread(images_directory + '/' + folder['name'] + '/' + image_name)
        image_grey = preprocess_image(image)
        region = find_biggest_region(image_grey)
        corners_count = count_corners(image_grey, False)
        lines_count = count_lines_in_skeleton(image_grey, False)
        regions.append(region)
        corners_counts.append(corners_count)
        lines_counts.append(lines_count)
        output_for_file = folder['name'] + ',' \
                              + str(np.median(region)) + ','\
                              + str(np.median(corners_count)) + ','\
                              + str(np.median(lines_count))
        output_for_files.append(output_for_file)
        print(output_for_file)
    output_for_category = folder['name'] + ',' \
                          + str(np.median(regions)) + ','\
                          + str(np.median(corners_counts)) + ','\
                          + str(np.median(lines_counts))
    output_for_categories.append(output_for_category)
    print('-----------------')
    print(output_for_category)
    print('-----------------')

output_file = open(output_file, 'w')
output_file2 = open(output_file2, 'w')

for data in output_for_categories:
	output_file.write(data + "\n") #output z cechami wszystkich zdjęć we wszystkich folderach

for data in output_for_files:
    output_file2.write(data + "\n") #output całego zbioru folders_with_files