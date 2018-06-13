print('Loading packages...')
import os
import os.path
import sys
import skimage.io as io
import skimage.color as color
import skimage.feature as feature
import skimage.measure as measure
import skimage.morphology as mo

print('Packages loaded.\n')


# functions to check path, save results, count features

def to_csv(folder_name, features_list):
    csv = str(folder_name) + "," + (",".join(str(x) for x in features_list))
    return csv


def check_input_path():
    try:
        db_dir = sys.argv[1]
        if (os.path.exists(db_dir)) == False:
            print("ERROR: Invalid input path.")
            sys.exit()
    except:
        print("ERROR: Missing input directory path.\n")
        sys.exit()

    return (db_dir)


def check_output_path():
    try:
        output_file = sys.argv[2]
        if (os.path.exists(output_file)):
            print("WARNING: File with that same name ({}) already exists.".format(output_file))
    except:
        print("ERROR: Missing output file path.\n")
        sys.exit()
    return (output_file)


def euclidean_distance(x, y):
    x1, y1 = x
    x2, y2 = y
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return distance


def preprocess_image(img, cropp_x, cropp_y, ratio, trim=False):
    if trim:
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
            dist = euclidean_distance(center, e.centroid)
            region = {"bbox": e.bbox, "bbox_area": e.bbox_area, "dist": dist, "area": e.area,
                      "convex_area": e.convex_area,
                      "eccentricity": e.eccentricity, "equivalent_diameter": e.equivalent_diameter,
                      "extent": e.extent, "major_axis_length": e.major_axis_length,
                      "minor_axis_length": e.minor_axis_length, "perimeter": e.perimeter, "solidity": e.solidity}
            regions.append(region)
    if len(regions) > 0:
        min_region = min(regions, key=lambda x: x["dist"])
        x1, y1, x2, y2 = min_region["bbox"]
        img = img[x1:x2, y1:y2]
        t = (
            min_region["area"] / min_region["bbox_area"] * 100,
            min_region["convex_area"] / min_region["bbox_area"] * 100,
            min_region["equivalent_diameter"] / min_region["bbox_area"] * 100, min_region["extent"],
            min_region["major_axis_length"] / min_region["bbox_area"] * 100,
            min_region["minor_axis_length"] / min_region["bbox_area"] * 100,
            min_region["perimeter"] / min_region["bbox_area"] * 100, min_region["solidity"],
            min_region["area"] / min_region["perimeter"] * 100)
        return (img, t)
    else:
        t = (min_area, 0, 0, 0, 0, 0, 0, 0, 0)
        return (img, t)


def count_corners(img):
    coords = feature.corner_peaks(feature.corner_harris(img))
    coords_subpix = feature.corner_subpix(img, coords)
    return len(coords_subpix)


def count_lines_in_skeleton(image):
    skeleton = mo.skeletonize(image < 1)
    lines = [item for sublist in skeleton for item in sublist]
    return lines.count(True)


# main loop works only when file is running directly
if __name__ == "__main__":
    db_dir = check_input_path()
    output_filename = check_output_path()
    print('\nOK - Checked input params.\n')
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
        folder_name = folders[folder]
        print(" Progress: {}% ({}/{})".format((folder / len(folders) * 100), folder, len(folders)))
        for image_name in images_names[folder]:
            img = io.imread(db_dir + '/' + folders[folder] + '/' + image_name)
            img = preprocess_image(img, 680, 620, 0.72,
                                   trim=True)  # set False when imgs different than those in "leafsnap-subset1"
            (changed_img, features_tuple) = extract_region_props(img, 2000)
            corners = count_corners(changed_img)
            skeleton_lines = count_lines_in_skeleton(changed_img)
            features_tuple = features_tuple + (corners, skeleton_lines)
            csv = to_csv(folder_name, features_tuple)
            output_file.write(csv + '\n')
    output_file.close()
    print('\nFINISHED - Features saved to file {}.\n'.format(output_filename))
