print('Loading packages...')
import os
import os.path
import sys
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.color as color
import skimage.data as data
import skimage.feature as feature
import skimage.io as io
import skimage.measure as measure
import skimage.morphology as mo
from scipy.signal import convolve2d
from skimage.draw import ellipse
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from scipy import ndimage as ndi
print('Packages loaded.\n')


def calculate_euclidean_distance(x, y):
    x1, y1 = x
    x2, y2 = y
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2)**0.5
    return distance

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

def to_csv(folder_name, features_list):
    csv = str(folder_name) + "," + (",".join(str(x) for x in features_list))
    return csv

def check_input_path():
    try:
        db_dir = sys.argv[1]
        if (os.path.exists(db_dir))==False:
            print ("ERROR: Invalid input path.")
            sys.exit()
    except:
        print("ERROR: Missing input directory path.\n")
        sys.exit()

    return(db_dir)

def check_output_path():
    try:
        output_file = sys.argv[2]
        if (os.path.exists(output_file)):
            print ("WARNING: File with that same name ({}) exists yet.".format(output_file))
    except:
        print("ERROR: Missing output file path.\n")
        sys.exit()
    return(output_file)

# main loop
if __name__ == "__main__":

    db_dir, folders ='', '' 
    db_dir=check_input_path()
    output_filename = check_output_path()
    print('\nOK - Checked input params.\n')

    # dictionaries for features:
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

    folders = list(os.listdir(db_dir))

    if (os.path.exists(db_dir + '/readme.txt')):
        folders.remove('readme.txt')

    images_names = []
    for folder in folders:
        l = list(os.listdir(db_dir + '/' + folder))
        images_names.append(l)

    output_file = open(output_filename, 'w')
    
    print("WORKING in progress... Please wait.")
    for folder in range(len(folders)):
        print (" Progress: {}/{}".format(folder, len(folders)))
        for image_name in images_names[folder]:
            img = io.imread(db_dir + '/' + folders[folder] + '/' + image_name)
            img = preprocess_image(img, 680, 620, 0.72)
            img, surface, convex_area, eccentricity, equivalent_diameter, extent, major_axis_length, \
            minor_axis_length, perimeter = extract_region_props(img, 2000)
    # powyższe cechy można zapisać jako features_tuple i następnie konwertować poprzez
    # komendę "features_list = list(features_tuple)"
    #
    # kolejne cechy - pozmieniać, bo są takie jak Pawła
    #         c1 = surface
    #         c2 = convex_area
    #         c3 = eccentricity
    #         c4 = equivalent_diameter
    #         c5 = extent
    #         c6 = major_axis_length
    #         c7 = minor_axis_length
    #         c8 = perimeter
    #         c9 = count_corners(img)
    #         c10 = len(feature.peak_local_max(img))
    #         c11 = detect_censure_scales(img, censure)
    #         features_list = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]
    #         csv = to_csv(folders[folder], features_list)
    #         output_file.write(csv + '\n')
    output_file.close()
    print('\nFINISHED - Features saved to file {}.\n'.format(output_filename))