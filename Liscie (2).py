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
from scipy import ndimage as ndi
from skimage import feature
import sys

db_dir = sys.argv[1]
output_dir = sys.argv[2]
print('\nInput parameters - ok!\n')

print('Loading libraries...')
from math import sqrt
from os import listdir, path
import numpy as np
import skimage.io as io
import skimage.color as color
import skimage.measure as measure
import skimage.feature as feature

print('Libraries loaded successfully!\n')

def calculate_euclidean_distance(x, y):
    x1, y1 = x
    x2, y2 = y
    dist = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist

def preprocess_image(img, cropp_x, cropp_y, ratio):
    img = img[:cropp_x, :cropp_y]
    img_grey = color.rgb2gray(img)
    img_grey = (img_grey < ratio)
    img_labeled = measure.label(img_grey)
    return img_labeled

def extract_region_props(img, min_area):
    center = (int(img.shape[0] / 2), int(img.shape[1] / 2))
    regions = []
    for i, e in enumerate(measure.regionprops(img)):
        if e.area > min_area:
            dist = calculate_euclidean_distance(center, e.centroid)
            region = {"bbox": e.bbox, "dist": dist, "area": e.area, "convex_area": e.convex_area, \
                      "eccentricity": e.eccentricity, "equivalent_diameter": e.equivalent_diameter, \
                      "extent": e.extent, "major_axis_length": e.major_axis_length, \
                      "minor_axis_length": e.minor_axis_length, "perimeter": e.perimeter}
            regions.append(region)
    if len(regions) > 0:
        min_region = min(regions, key=lambda x: x["dist"])
        x1, y1, x2, y2 = min_region["bbox"]
        img = img[x1:x2, y1:y2]
        t = (img, min_region["area"], min_region["convex_area"], min_region["eccentricity"], \
             min_region["equivalent_diameter"], min_region["extent"], min_region["major_axis_length"], \
             min_region["minor_axis_length"], min_region["perimeter"])
        return t
    else:
        t = (img, min_area, 0, 0, 0, 0, 0, 0, 0)
        return t

def count_corners(img):
    coords = feature.corner_peaks(feature.corner_harris(img))
    coords_subpix = feature.corner_subpix(img, coords)
    return len(coords_subpix)

def detect_censure_scales(img, censure):
    censure.detect(img)
    return (len(censure.scales))

def to_csv(folder_name, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11):
    csv = str(folder_name) + "," \
          + str(c1) + "," \
          + str(c2) + "," \
          + str(c3) + "," \
          + str(c4) + "," \
          + str(c5) + "," \
          + str(c6) + "," \
          + str(c7) + "," \
          + str(c8) + "," \
          + str(c9) + "," \
          + str(c10) + "," \
          + str(c11)
    return csv

all_surfaces = {}
all_convex_areas = {}
all_eccentricities = {}
all_equivalent_diameters = {}
all_extents = {}
all_major_axis_lengths = {}
all_minor_axis_lengths = {}
all_perimeters = {}
all_corners = {}
all_peaks = {}
all_censure_scales = {}
censure = feature.CENSURE()

folders = list(listdir(db_dir))

if (path.exists(db_dir + '/readme.txt')):
    folders.remove('readme.txt')

images_names = []
for folder in folders:
    l = list(listdir(db_dir + '/' + folder))
    images_names.append(l)

output_file = open("output.csv", 'a')
for folder in range(len(folders)):
    for image_name in images_names[folder]:
        img = io.imread(db_dir + '/' + folders[folder] + '/' + image_name)
        img = preprocess_image(img, 680, 620, 0.72)
        img, surface, convex_area, eccentricity, equivalent_diameter, extent, major_axis_length, \
        minor_axis_length, perimeter = extract_region_props(img, 2000)
		#kolejne cechy - pozmieniać, bo są takie jak Pawła
        c1 = surface
        c2 = convex_area
        c3 = eccentricity
        c4 = equivalent_diameter
        c5 = extent
        c6 = major_axis_length
        c7 = minor_axis_length
        c8 = perimeter
        c9 = count_corners(img)
        c10 = len(feature.peak_local_max(img))
        c11 = detect_censure_scales(img, censure)
        csv = to_csv(folders[folder], c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)
        output_file.write(csv + '\n')
output_file.close()
print('Extracted features has been saved to file: ', output_dir, '\n')