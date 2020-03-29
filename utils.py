import tensorflow as tf

base_network_init = {
    'inceptionv3': tf.keras.applications.inception_v3.InceptionV3,
    'vgg16': tf.keras.applications.vgg16.VGG16,
    'resnet50': tf.keras.applications.resnet50.ResNet50
}

preprocessed_input = {
    'inceptionv3': tf.keras.applications.inception_v3.preprocess_input,
    'vgg16': tf.keras.applications.vgg16.preprocess_input,
    'resnet50': tf.keras.applications.resnet50.preprocess_input
}

base_sizes = {
    'inceptionv3': (299, 299),
    'vgg16': (224, 224),
    'resnet50': (224, 224)
}

base_output_pooling = {
    'average': tf.keras.layers.GlobalAveragePooling2D,
    'max': tf.keras.layers.GlobalMaxPool2D
}
