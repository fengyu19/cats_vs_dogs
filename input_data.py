
import numpy as np
import os
import tensorflow as tf

train_dir = '/home/fengyu/python_workspace/cats_vs_dogs/picData/train/train/'
test_dir = '/home/fengyu/python_workspace/cats_vs_dogs/picData/test/test/'

def get_files(file_dir):
    '''
    :param file_dir:
        fire directory
    :return:
        list of images and labels
    '''
    cats = []
    cats_label = []
    dogs = []
    dogs_label = []
    for file in os.listdir(train_dir):
        name = file.split('.')
        if name[0] == 'cat':
            cats_label.append(0)
            cats.append(file_dir + file)
        else:
            dogs_label.append(1)
            dogs.append(file_dir + file)
    print 'there are %d cats and %d dogs' %(len(cats_label), len(dogs_label))

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((cats_label, dogs_label))
    # print image_list, label_list

    temp = np.array([image_list, label_list])
    # print temp
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


# image_list, label_list = get_files(train_dir)
# print image_list
def get_testfiles(file_dir):
    '''
    :param file_dir:
        fire directory
    :return:
        list of images and labels
    '''
    animal = []
    for file in os.listdir(test_dir):
        animal.append(file_dir + file)
    print 'there are %d animal' %(len(animal))

    image_list = np.hstack((animal))
    # print image_list, label_list

    temp = np.array([image_list])
    # print temp
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])

    return image_list




def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''

    :param image: list type
    :param label: list type
    :param image_W: image width
    :param image_H: image height
    :param batch_size: batch size
    :param capacity: the capacity of the queue
    :return:
        image batch: 4D tensor [batch size, width, height, 3], dtype=tf.float32
        label batch:  1D tensor [batch size] dtype=tf.int32
    '''
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=32, capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch


# test

# import matplotlib.pyplot as plt
#
# BATCH_SIZE = 5
# CAPACITY = 128
# IMG_W = 208
# IMG_H = 208
# image_list, label_list = get_files(train_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#     i = 0
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     try:
#         while not coord.should_stop() and i<1:
#
#             img, label = sess.run([image_batch, label_batch])
#
#             for j in np.arange(BATCH_SIZE):
#                 print 'label: %d' %label[j]
#                 plt.imshow(img[j, :, :, :])
#                 plt.show()
#             i = i + 1
#
#     except tf.errors.OutOfRangeError:
#         print 'Done!'
#     finally:
#         coord.request_stop()
#     coord.join(threads)




