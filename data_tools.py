import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split

np.random.seed(666)


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label


def preprocess_image(image, w, h):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [w, h])
    image = (image - 127.5) / 128  # normalize to [-1,1] range
    return image


def load_and_preprocess_image(path, w=112, h=112):
    image = tf.io.read_file(path)
    return preprocess_image(image, w, h)


def load_and_preprocess_from_path_label(path, label, img_size):
    return load_and_preprocess_image(path, w=img_size[0], h=img_size[1]), label


def augmentation(x, y, size=[112, 112]):
    x = tf.image.resize_with_crop_or_pad(
        x, size[0] + 8, size[1] + 8)
    x = tf.image.random_crop(x, [size[0], size[1], 3])
    x = tf.image.random_flip_left_right(x)
    return x, y


def get_data(datdir):
    categories = os.listdir(datdir)
    label_to_index = dict((name, index) for index, name in enumerate(categories))
    all_image_paths = []
    all_image_labels = []
    for category in categories:
        path = os.path.join(datdir, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            all_image_paths.append(img_path)
            all_image_labels.append(label_to_index[category])
    print("图片数量:", len(all_image_paths))
    print("类别数量:", len(categories))
    r = np.random.permutation(len(all_image_paths))
    all_image_paths = np.asarray(all_image_paths)[r]
    all_image_labels = np.asarray(all_image_labels)[r]
    return all_image_paths, all_image_labels


def get_train_test(all_image_paths, all_image_labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(all_image_paths, all_image_labels, shuffle=True,
                                                        test_size=test_size, random_state=666)
    return X_train, X_test, y_train, y_test


def get_kfold(all_image_paths, all_image_labels, K=5):
    kf = KFold(n_splits=K, shuffle=True)
    KFTrain_X = []
    KFTest_X = []
    KFTrain_Y = []
    KFTest_Y = []
    for train, test in kf.split(all_image_paths):
        KFTrain_X.append(all_image_paths[train])
        KFTest_X.append(all_image_paths[test])
        KFTrain_Y.append(all_image_labels[train])
        KFTest_Y.append(all_image_labels[test])
    return KFTrain_X, KFTrain_Y, KFTest_X, KFTest_Y


def dats_pipleline(x, y, BATCH_SIZE=32, is_train=True, aug=True, img_size=[112, 112]):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(lambda x, y: load_and_preprocess_from_path_label(x, y, img_size))
    if is_train:
        ds = ds.shuffle(buffer_size=1000)
        if aug:
            ds = ds.map(lambda x, y: augmentation(x, y, img_size))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


if __name__ == '__main__':
    X, Y = get_data("data/dataset_kaggledogvscat/train")
    train_X, train_Y, test_X, test_Y = get_kfold(X, Y)
    print(np.shape(test_Y))
    train_db = dats_pipleline(test_X[0], test_Y[0], img_size=[160, 160])
    for step, (x, y) in enumerate(train_db):
        print(np.shape(x))
