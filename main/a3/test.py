import cv2
import numpy as np
import os
import shutil

from keras import models, layers, backend
from models.vgg16 import VGG16
from tqdm import tqdm


def get_pairs():
    file_path = "../../data/data_a3/testPairs"

    with open(file_path) as f:
        content = f.readlines()

    pairs = []
    for line in content:
        index = line.find(' ')
        pairs.append([line[:index], line[index + 1:-1]])

    return pairs


def crop_images(path):
    files = os.listdir(path)

    for file in files:
        img_path = os.path.join(path, file)
        image = cv2.imread(img_path)
        crop_img = image[13:13 + 224, 13:13 + 224]
        cv2.imwrite(img_path, crop_img)


vgg16 = VGG16(include_top=False,
              weights=None,
              input_shape=(224, 224, 3))

middle = models.Sequential()
middle.add(vgg16)

middle.add(layers.Flatten())
middle.add(layers.Dense(4096, activation='relu', name='fc1'))
middle.add(layers.Dense(4096, activation='relu', name='fc2'))
middle.add(layers.Lambda(lambda x: backend.l2_normalize(x, axis=1)))
middle.load_weights("../../weights/a3_best_model.hdf5", by_name=True)

test_dir_path = "../../data/data_a3/test"
cropped_test_dir_path = "../../data/data_a3/cropped_test"
os.makedirs(cropped_test_dir_path, exist_ok=True)
for file in os.listdir(test_dir_path):
    shutil.copy(os.path.join(test_dir_path, file), os.path.join(cropped_test_dir_path, file))

crop_images(cropped_test_dir_path)

original_files = os.listdir(cropped_test_dir_path)
desired_size = 224
for file in tqdm(original_files):
    im_pth = os.path.join(cropped_test_dir_path, file)
    im = cv2.imread(im_pth)
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = 1  # float(desired_size)/max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    cv2.imwrite(os.path.join(cropped_test_dir_path, file), new_im)

pairs = get_pairs()

for name1, name2 in tqdm(pairs):
    image1 = cv2.imread(os.path.join(cropped_test_dir_path, name1))
    image2 = cv2.imread(os.path.join(cropped_test_dir_path, name2))

    embedding_original_1 = middle.predict(np.expand_dims(image1 / 255., axis=0))[0]
    embedding_hflip_1 = middle.predict(np.expand_dims(np.fliplr(image1) / 255., axis=0))[0]
    average_embedding_1 = (embedding_original_1 + embedding_hflip_1) / 2.

    embedding_original_2 = middle.predict(np.expand_dims(image2 / 255., axis=0))[0]
    embedding_hflip_2 = middle.predict(np.expand_dims(np.fliplr(image2) / 255., axis=0))[0]
    average_embedding_2 = (embedding_original_2 + embedding_hflip_2) / 2.

    dist = np.sum((average_embedding_1 - average_embedding_2) ** 2)
    dist = np.sqrt(dist)

    file = open("../../data/data_a3/results.txt", 'a')
    result = 1 - dist

    s = str(result) + '\n'
    file.write(s)
    file.close()
