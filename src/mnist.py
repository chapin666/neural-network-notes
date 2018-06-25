import numpy as np
import tensorflow as tf
import pandas as pd

# load data
train = pd.read_csv('../data/train.csv')
images = train.iloc[:, 1:].values;
labels_flat = train.iloc[:,0].values.ravel()

# input processing
images = images.astype(np.float)

images = np.multiply(images, 1.0 / 255.0)
print('输入数据的数量: (%g, %g)'  % images.shape)

image_size = images.shape[1]
print('输入数据的纬度 => {0}'.format(image_size))

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print('图片的长 => {0}\n图片的高 => {1}'.format(image_width, image_height))

x = tf.placeholder('float', shape=[None, image_size]);
labels_count = np.unique(labels_flat).shape[0]

print('结果的种类 => {0}'.format(labels_count))
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)
print('结果的数量: ({0[0]}, {0[1]})'.format(labels.shape))

y = tf.placeholder('float', shape=[None, labels_count])

VALIDATION_SIZE = 2000

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

batch_size = 100
n_batch = int(len(train_images)/batch_size)
