import os
import shutil
import cv2


def copy_images_to_dir(source_path, target_path):
    """
    Copy images from the same class to one directory with the class name
    (because VGG16 interprets each directory as one class and all images in this directory are from that class)
    """

    files = sorted(os.listdir(source_path))
    for file in files:
        name = file[:file.rfind("_")]
        directory = target_path + name
        if not os.path.isdir(directory):
            os.mkdir(directory)
        shutil.copyfile(source_path + file, os.path.join(directory, file))


def remove_dirs_with_one_image(path):
    dirs = os.listdir(path)
    for dir in dirs:
        dir_path = os.path.join(path, dir)
        if not len(os.listdir(dir_path)) > 1:
            shutil.rmtree(dir_path)


def crop_images(path):
    dirs = [os.path.join(path, d) for d in os.listdir(path)]

    for dir in dirs:
        files = os.listdir(dir)
        for file in files:
            img_path = os.path.join(dir, file)
            image = cv2.imread(img_path)
            crop_img = image[13:13 + 224, 13:13 + 224]
            cv2.imwrite(img_path, crop_img)


def main():
    source_path = "../../data/data_a3/train/"
    target_path = "../../data/data_a3/cropped_train/"
    copy_images_to_dir(source_path, target_path)
    remove_dirs_with_one_image(target_path)
    crop_images(target_path)


if __name__ == "__main__":
    main()
