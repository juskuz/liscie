print('Loading packages...')
import os.path
import sys
import skimage.io as io
import numpy as np
print('Packages loaded.\n')
print("Loading functions to extact...")
import extract_features
print('Functions loaded.\n')

from sklearn.externals import joblib

def save_results(results_list):
    file = open("matching.txt", "w")
    for line in results_list:
        file.write(line)
    print ("Results saved to file matching.txt.")

# main loop
if __name__ == "__main__":
    db_dir = extract_features.check_input_path()
    learned_classifier_name="trained_model.pkl"
    extraTreesClassifier = joblib.load("trained_model.pkl")
    folders = list(os.listdir(db_dir))
    if (os.path.exists(db_dir + '/readme.txt')):
        folders.remove('readme.txt')
    images_names = []
    results_list = []
    mistakes = 0
    correct = 0
    for folder in folders:
        l = list(os.listdir(db_dir + '/' + folder))
        images_names.append(l)
    for folder in range(len(folders)):
        folder_name = folders[folder]
        for image_name in images_names[folder]:
            img = io.imread(db_dir + '/' + folders[folder] + '/' + image_name)
            img = extract_features.preprocess_image(img, 680, 620, 0.72, trim=True) # set trim=False if imgs size different than in 'leafsnap-subset1'
            (changed_img, features_tuple) = extract_features.extract_region_props(img, 2000)
            corners = extract_features.count_corners(changed_img)
            skeleton_lines = extract_features.count_lines_in_skeleton(changed_img)
            features_tuple = features_tuple + (corners, skeleton_lines)
            features_list = list(features_tuple)
            data = np.array([features_list])
            # print(folder_name, image_name, extraTreesClassifier.predict(data)[0])
            result = (image_name + extraTreesClassifier.predict(data)[0] +"\n")
            results_list.append(result)
            if (folder_name != extraTreesClassifier.predict(data)[0]):
                mistakes+=1
            else:
                correct+=1
    print ("Correct: {}\nIncorrect: {}\nPercent of correct matches: {}%".format(correct, mistakes, correct/(correct+mistakes)*100))
    save_results(results_list)
