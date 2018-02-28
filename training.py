import os
import numpy as np
import tensorflow as tf
import model
import input_data

n_classes = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 24
CAPACITY = 2000
MAX_STEP = 15000
learning_rate = 0.0005

def run_training():
    train_dir = '/home/fengyu/python_workspace/cats_vs_dogs/picData/train/train/'
    logs_traing_dir = '/home/fengyu/python_workspace/cats_vs_dogs/logs/'

    train, train_label = input_data.get_files(train_dir)

    train_batch, train_label_batch = input_data.get_batch(train, train_label,IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    train_logits = model.inference(train_batch, BATCH_SIZE, n_classes)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.trainning(train_loss, learning_rate)
    train_accuracy = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()

    train_writer = tf.summary.FileWriter(logs_traing_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_accuracy = sess.run([train_op, train_loss,train_accuracy])

            if step%50 == 0:
                print 'Step %d, train loss = %.2f, accuracy = %.2f' %(step, tra_loss, tra_accuracy)
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step%2000 == 0 or (step+1) == MAX_STEP:
                check_point_path = os.path.join(logs_traing_dir, 'model.ckpt')
                saver.save(sess, check_point_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print 'Done training -- epoch limit reached'
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

# run_training()

from PIL import Image
import matplotlib.pyplot as plt

def get_one_image(train):
    '''

    :param train:
    :return: ndarray
    '''

    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]

    image = Image.open(img_dir)
    plt.imshow(image)


    image = image.resize([208,208])
    image = np.array(image)
    return image

def evaluate_one_image():

    # train_dir = '/home/fengyu/python_workspace/cats_vs_dogs/picData/train/train/'
    train_dir = '/home/fengyu/python_workspace/cats_vs_dogs/picData/test/test/'
    # train, train_label = input_data.get_files(train_dir)
    train = input_data.get_testfiles(train_dir)
    image_array = get_one_image(train)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        n_classes = 2

        image = tf.cast(image_array, tf.float32)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, n_classes)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[208,208,3])

        logs_traing_dir = '/home/fengyu/python_workspace/cats_vs_dogs/logs/'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print 'reading chechpoints...'
            ckpt = tf.train.get_checkpoint_state(logs_traing_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print 'Loading success, global_step is %s' %global_step
            else:
                print 'no checkpoint file found'

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)

            if max_index == 0:
                print 'this is a cat with possibility %.6f' %prediction[:,0]
            else:
                print 'this is a dog with possibility %.6f' % prediction[:, 1]
            plt.show()

evaluate_one_image()

