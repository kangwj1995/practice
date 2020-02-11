import tensorflow as tf
import numpy as np
import glob
import random
from matplotlib import pyplot as plt
from PIL import Image


def _bytes_feature(value):
    """ Returns a bytes_list from a string/byte"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """ Returns a float_list from a float/double """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """ Returns a int64_list from a bool/enum/int/uint """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _image_as_bytes(imagefile):
    image = np.array(Image.open(imagefile))
    image_raw = image.tostring()
    return image_raw

def make_example(img, lab):
    """ TODO: Return serialized Example from img, lab """
    feature = {'encoded': _bytes_feature(img), 'label': _int64_feature(lab)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def write_tfrecord(test_imagedir, train_imagedir, test_datadir, train_datadir, val_datadir): # Put a new input 'process', in order to make validation set easier.
    """ TODO: write a tfrecord file containing img-lab pairs
        imagedir: directory of input images
        datadir: directory of output a tfrecord file (or multiple tfrecord files) """
    test_writer = tf.python_io.TFRecordWriter(test_datadir+'/test.tfrecord')
    for i in range(10):
        filenames = glob.glob(test_imagedir + '/' + str(i) + '/*')
        for j in range(len(filenames)):
            filename = filenames[j]
            lab = i
            example = make_example(_image_as_bytes(filename), lab)
            test_writer.write(example)
    test_writer.close()

    train_writer = tf.python_io.TFRecordWriter(train_datadir + '/train.tfrecord')
    val_writer = tf.python_io.TFRecordWriter(val_datadir + '/val.tfrecord')
    array = []
    labels = []
    for i in range(10):
        filenames = glob.glob(train_imagedir + '/' + str(i) + '/*')
        lab = i
        for j in range(len(filenames)):
            array.append(filenames[j])
            labels.append(lab)
    merged = np.rec.fromarrays([array, labels], names=('filename', 'label'))
    random.shuffle(merged)

    for i in range(len(merged)):
        img_raw, label = merged[i]
        example = make_example(_image_as_bytes(img_raw), label)
        if i > 0.25*len(merged) :
            train_writer.write(example)
        else :
            val_writer.write(example)
    train_writer.close()
    val_writer.close()


def read_tfrecord(folder, batch=100, epoch=1):
    """ TODO: read tfrecord files in folder, Return shuffled mini-batch img,lab pairs
    img: float, 0.0~1.0 normalized
    lab: dim 10 one-hot vectors
    folder: directory where tfrecord files are stored in
    epoch: maximum epochs to train, default: 1 """

    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(tf.train.string_input_producer(glob.glob(folder), num_epochs=epoch))
    key_to_feature = {'encoded': tf.FixedLenFeature([], tf.string, default_value=''), 'label': tf.FixedLenFeature([], tf.int64, default_value=0)}
    features = tf.parse_single_example(serialized_example, features=key_to_feature)
    img = tf.reshape(tf.cast(tf.decode_raw(features['encoded'], tf.uint8), tf.float32)/255.0, [28, 28, 1])
    lab = tf.one_hot(tf.cast(features['label'], tf.int32), 10)
    img, lab = tf.train.shuffle_batch([img, lab], batch_size=batch, capacity=10*batch, num_threads=1, min_after_dequeue=2*batch)

    return img, lab


