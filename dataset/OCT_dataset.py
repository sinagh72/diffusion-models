"""
OCT_dataset.py: OCT Dataset script
__author: Sina Gholami
__update: add comments
__update_date: 7/5/2024
__note: working with python <= 3.10
"""
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
from PIL import Image
from natsort import natsorted
import subsetsum as sb

from utils.utils import find_key_by_value


class OCTDataset(Dataset):

    def __init__(self, img_type="L", transform=None, img_paths=None, **kwargs):
        self.transform = transform  # transform functions
        self.img_type = img_type  # the type of image L, R
        self.img_paths = img_paths
        self.kwargs = kwargs

    def __getitem__(self, index):
        img_path, (label, _) = self.img_paths[index]
        img_path = img_path.replace("\\", "/")  # fixing the path for windows os
        img_view = self.load_img(img_path)  # return an image
        if self.transform is not None:
            img_view = self.transform(img_view)
        return img_view, label

    def __len__(self):
        return len(self.img_paths)

    def load_img(self, img_path):
        img = Image.open(img_path).convert(self.img_type)
        return img


def get_kermany_imgs(data_dir: str, **kwargs):
    # make sure the sum of split is 1
    split = kwargs["split"]
    classes = kwargs["classes"]
    if "tokenizer" in kwargs:
        tokenizer = kwargs["tokenizer"]
    else:
        tokenizer = "-"
    split = np.array(split)
    assert (split.sum() == 1)

    img_paths = [[] for _ in split]
    img_filename_list = []
    path = os.listdir(os.path.join(data_dir))
    # filter out the files not inside the classes
    for c in classes.keys():
        img_filename_list += list(filter(lambda k: c in k, path))
    for img_file in img_filename_list:  # iterate over each class
        img_file_path = os.path.join(data_dir, img_file)
        # sort lexicographically the img names in the img_file directory
        img_names = natsorted(os.listdir(img_file_path))
        img_dict = {}
        # patient-wise dictionary
        for img in img_names:
            img_num = img.split(tokenizer)[2]  # the number associated to each img
            img_name = img.replace(img_num, "")
            if img_name in img_dict:
                img_dict[img_name] += [img_num]
            else:
                img_dict[img_name] = [img_num]
        selected_keys = set()  # Keep track of images that have already been added
        copy_split = split.copy()
        for i, percentage in enumerate(copy_split):
            # create a list of #visits of clients that has not been selected
            num_visits = [len(img_dict[key]) for key in img_dict if key not in selected_keys]
            total_imgs = sum(num_visits)
            selected_num = math.ceil(total_imgs * (percentage))
            subset = []
            for solution in sb.solutions(num_visits, selected_num):
                # `solution` contains indices of elements in `nums`
                subset = [i for i in solution]
                break
            keys = [key for key in img_dict if key not in selected_keys]
            for idx in subset:
                selected_subset = [(img_file_path + "/" + keys[idx] + count,
                                    get_class(img_file_path + keys[idx], classes)) for count in img_dict[keys[idx]]]
                img_paths[i] += selected_subset
                selected_keys.add(keys[idx])  # Mark this key as selected
            if len(copy_split) > i + 1:
                for j in range(i + 1, len(copy_split)):
                    copy_split[j] += percentage / (len(copy_split) - (i + 1))
    return img_paths


def get_srinivasan_imgs(data_dir: str, **kwargs):
    """
    :param data_dir: str, the path to the dataset
    :param kwargs:
        - ignore_folders (np.array): indices of files to ignore
        - classes (dict): {NORMAL:0, AMD:1, DME:2}
    :return:
    """
    classes = kwargs["classes"]
    img_filename_list = []
    all_files = natsorted(os.listdir(os.path.join(data_dir)))
    for c in classes:
        img_filename_list += [file for file in all_files if c in file]
    imgs_path = []
    for img_file in img_filename_list:
        if any(item == int(img_file.replace("AMD", "").replace("NORMAL", "").replace("DME", ""))
               for item in kwargs["ignore_folders"]):
            continue
        folder = os.path.join(data_dir, img_file, "TIFFs/8bitTIFFs")
        imgs_path += [(os.path.join(folder, id), get_class(os.path.join(folder, id), kwargs["classes"]))
                      for id in os.listdir(folder)]
    return imgs_path


def get_oct500_imgs(data_dir: str, **kwargs):
    assert round(sum((kwargs["split"]))) == 1
    classes = kwargs["classes"]
    mode = kwargs["mode"]
    split = kwargs["split"]

    df = pd.read_excel(os.path.join(data_dir, "Text labels.xlsx"))
    # whether to merge any classes
    if "merge" not in kwargs or kwargs["merge"] is None:
        kwargs["merge"] = {}
    for key, val in kwargs["merge"].items():
        for old_class in val:
            df['Disease'] = df['Disease'].replace(old_class, key)
    img_paths = []
    for c in classes.keys():
        temp_path = []
        disease_ids = df[df["Disease"] == c]["ID"].sort_values().tolist()
        train, val, test = split_oct500(disease_ids, split)
        if mode == "train":
            temp_path += get_optovue(train, data_dir, class_label=(classes[c], c), filter_img=kwargs["filter_img"],
                                     m6_range=(160, 240), m3_range=(100, 180))
        elif mode == "val":
            temp_path += get_optovue(val, data_dir, class_label=(classes[c], c), filter_img=kwargs["filter_img"],
                                     m6_range=(160, 240), m3_range=(100, 180))
        elif mode == "test":
            temp_path += get_optovue(test, data_dir, class_label=(classes[c], c), filter_img=kwargs["filter_img"],
                                     m6_range=(160, 240), m3_range=(100, 180))
        img_paths += temp_path
    return img_paths


def split_oct500(total_ids: list, train_val_test: tuple):
    """
    Divides config into train, val, test
    :param total_ids: list of patients ids
    :param train_val_test: (train split, val split, test split) --> the sum should be 1
    """
    total_length = len(total_ids)

    train_idx = math.floor(total_length * train_val_test[0])
    val_idx = math.floor(total_length * train_val_test[1])

    train_data = total_ids[:train_idx]
    val_data = total_ids[train_idx:train_idx + val_idx]
    test_data = total_ids[train_idx + val_idx:]

    return train_data, val_data, test_data


def get_optovue(list_ids, data_dir, class_label, filter_img=True,
                m8_range=(220, 300), m6_range=(160, 240), m3_range=(100, 180)):
    """
    retrieve optovue data
    :param list_ids: list,
    :param data_dir: str,
    :param class_label: str,
    :param filter_img: bool, if True, only foveal images will be loaded
    :m8_range: tuple, range of 8mm-OCT slices to load in case filter_img = True
    :m6_range: tuple, range of 6mm-OCT slices to load in case filter_img = True
    :m3_range: tuple, range of 3mm-OCT slices to load in case filter_img = True
    """
    img_paths = []
    for idd in list_ids:
        file_path = os.path.join(data_dir, "OCT", str(idd))
        for img in os.listdir(file_path):
            if filter_img and (("6mm" in data_dir and m6_range[0] <= int(img[:-4]) <= m6_range[1]) or
                               ("3mm" in data_dir and m3_range[0] <= int(img[:-4]) <= m3_range[1]) or
                               ("8mm" in data_dir and m8_range[0] <= int(img[:-4] <= m8_range[1]))):
                img_paths.append((os.path.join(file_path, img), class_label))
            elif filter_img is False:
                img_paths.append((os.path.join(file_path, img), class_label))
    return img_paths


def get_nur_dataset(data_dir, csv_filename, classes, split=(0.8, 0.1, 0.1)):
    """

    :param data_dir: the root the dataset
    :param csv_filename: the name of the csv file containing the annotation
    :param classes: the classes in dict {"NORMAL": 0, ...}
    :param split: split percentage
    :return: three lists --> train list, validation list, test list
    """
    # Check if the split ratios sum to 1
    if sum(split) != 1:
        raise ValueError("Split ratios must sum to 1")

    # Load CSV file using Pandas
    df = pd.read_csv(os.path.join(data_dir, csv_filename))
    # Initialize lists to hold the split data
    train_data, validation_data, test_data = [], [], []
    for c in classes.keys():
        # Filter the dataframe for the current class
        class_df = df[df['Class'] == c]
        # Group by 'Patient ID'
        patients = class_df['Patient ID'].unique()
        # Split patients into train, validation, and test sets
        train_patients, test_patients = train_test_split(patients, train_size=split[0],
                                                         test_size=split[1] + split[2], random_state=42)
        validation_patients, test_patients = train_test_split(test_patients,
                                                              train_size=split[1] / (split[1] + split[2]),
                                                              test_size=split[2] / (split[1] + split[2]),
                                                              random_state=42)

        # Go through the dataframe and add the tuple (directory, label) to the corresponding list
        for _, row in class_df.iterrows():
            patient_id = row['Patient ID']
            data_tuple = (os.path.join(data_dir, row['Directory']),
                          get_class(classes=classes, img_name=row['Directory'].split("/")[-1]))

            if patient_id in train_patients:
                train_data.append(data_tuple)
            elif patient_id in validation_patients:
                validation_data.append(data_tuple)
            elif patient_id in test_patients:
                test_data.append(data_tuple)

    return train_data, validation_data, test_data


def get_waterloo_dataset(data_dir, split, classes):
    split = np.array(split)
    assert (split.sum() == 1)

    img_paths = [[] for _ in split]
    file_names = [f for f in os.listdir(data_dir) if f in classes.keys()]
    # filter out the files not inside the classes
    for file in file_names:
        imgs = os.listdir(os.path.join(data_dir, file))
        c = 0
        for i, percentage in enumerate(split):
            next_c = c + math.ceil(len(imgs) * percentage)
            # Adjust for potential overshoot in the last iteration
            if i == len(split) - 1:
                next_c = len(imgs)
            img_paths[i] += [(os.path.join(data_dir, file, img), (classes[file], file)) for img in imgs[c:next_c]]
            c = next_c
    return img_paths


def get_UIC_DR_imgs(data_dir: str, **kwargs):
    classes = kwargs["classes"]
    if "merge" not in kwargs or kwargs["merge"] is None:
        kwargs["merge"] = {}
    # for key, val in kwargs["merge"].items():
    #     for old_class in val:
    #         classes.replace(old_class, key)+yhhh
    img_paths = [[], [], []]
    for i, f in enumerate(["train", "val", "test"]):
        for category in classes:
            for patient in os.listdir(os.path.join(data_dir, f, category)):
                for img in os.listdir(os.path.join(data_dir, f, category, patient)):
                    img_paths[i] += [(os.path.join(data_dir, f, category, patient, img), (classes[category], category))]
    return img_paths


def get_Mario_imgs(root, train_csv, val_csv, classes, column='image'):
    train_df = pd.read_csv(os.path.join(root, train_csv))
    val_df = pd.read_csv(os.path.join(root, val_csv))
    train_data = [(os.path.join(root, "train", row[column]), (row['label'], find_key_by_value(classes, row['label']))) for _, row in train_df.iterrows()]
    val_data = [(os.path.join(root, "val", row[column]), (-1, "None")) for _, row in val_df.iterrows()]

    return train_data, val_data


def get_class(img_name, classes: dict):
    """
    returns the label and category of the image
    ex: 1, AMD
    ex: 0, NORMAL
    """
    for c in classes.keys():
        if c.upper() in img_name.upper():
            return classes[c], c
