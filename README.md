# We5fromBIET
Source code
   . Init.py -
               plant_disease_classification import (
            datagenerator,
             classifier
             )
            from plant_disease_classification.dataset import DataSet

Main.py –
      
             def main(train, classify, help):
             if (help):
             print(help_message)
             sys.exit(0)
             else:
             if (train):
            iteration = click.prompt(‘Iteration count for training model’, type=int)
            trainer.train(num_iteration=iteration)
            else:
image_file_path = click.prompt(‘Image file path that is going to be classified’,                type=str)
            classifier.classify(file_path=image_file_path)


Classifier.py –

Defclassify(file_path=’plant_disease_classification/datasets/test/0a02f9b47e8082558fa257092f0cedee.jpg’):
    print(file_path)
    images = []
    image = cv2.imread(file_path)
    # Resizing the image to our desired size and
    # preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype(‘float32’)
    images = np.multiply(images, 1.0 / 255.0)
    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    X_batch = images.reshape(1, image_size, image_size, num_channels)
    graph = tf.get_default_graph()
    y_pred = graph.get_tensor_by_name(“y_pred:0”)
    # Let’s feed the images to the input placeholders
    x= graph.get_tensor_by_name(“x:0”)
    y_true = graph.get_tensor_by_name(“y_true:0”)
    y_test_images = np.zeros((1, 38))
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = session.run(y_pred, feed_dict=feed_dict_testing)
    print(result)




Datagenerator.py –

image_paths = []
    while True:
        line = process.stdout.readline()
        if line != ‘’:
            filename = os.path.basename(line)
            path = os.path.join(os.path.dirname(line), filename)
            image_paths.append(path)
        else:
            break
    return image_paths

def load_train_data(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    class_array = []
    extension_list = (‘*.jpg’, ‘*.JPG’)

    print(‘Going to read training images’)
    for fields in classes:   
        index = classes.index(fields)
        print(‘Now going to read {} files (Index: {})’.format(fields, index))
        for extension in extension_list:
            path = os.path.join(train_path, fields, extension)
            files = glob.glob(path)
            for fl in files:
                image = cv2.imread(fl)
                image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
                image = image.astype(np.float32)
                image = np.multiply(image, 1.0 / 255.0)
                images.append(image)
                label = np.zeros(len(classes))
                label[index] = 1.0
                labels.append(label)
                flbase = os.path.basename(fl)
                img_names.append(flbase)
                class_array.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    class_array = np.array(class_array)
    return images, labels, img_names, class_array

def read_train_sets(train_path, image_size, classes, validation_size):
    data_set = DataSet()

    images, labels, img_names, class_array = load_train_data(train_path, image_size, classes)
    images, labels, img_names, class_array = shuffle(images, labels, img_names, class_array)  

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])
validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = class_array[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = class_array[validation_size:]

    data_set.train = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_set.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

    return data_set

Dataset.py –

import numpy as np

class DataSet(object):

    def __init__(self, *args):
        if len(args) > 3:
            self._num_examples = args[0].shape[0]
            self._images = args[0]
            self._labels = args[1]
            self._img_names = args[2]
            self._cls = args[3]
        self._epochs_done = 0
        self._index_in_epoch = 0
        self.train = None
        self.valid = None

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        “””Return the next `batch_size` examples from this data set.
        Args:
          batch_size (int):
            Size of batch.
        Returns:
           Tuples of images, labels, image names and class. 
        “””

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]

Trainer.py –

import os
import tensorflow as tf
import datagenerator

# Just disables the warning, doesn’t enable AVX/FMA
os.environ[‘TF_CPP_MIN_LOG_LEVEL’] = ‘2’

train_path = os.path.join(‘plant_disease_classification’, ‘datasets/train’)
classes = os.listdir(train_path)
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3

batch_size = 32

def create_y_true():
    # labels
    y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes], name=’y_true’)
    y_true_cls = tf.math.argmax(y_true, axis=1)
    return y_true, y_true_cls

def get_data():
    # Load and read training data
    data = datagenerator.read_train_sets(train_path, img_size, classes, validation_size)
    print(‘Complete reading input data. Will Now print a snippet of it’)
    print(‘Number of files in Training-set:\t\t{}’.format(len(data.train.labels)))
    print(‘Number of files in Validation-set:\t{}’.format(len(data.valid.labels)))
    return data

# Load all the training and validation images and labels into memory
# using openCV and use that during training
data = get_data()

session = tf.compat.v1.Session()
x = tf.compat.v1.placeholder(tf.float32,
                             shape=[None, img_size, img_size, num_channels],
                             name=’x’)

def create_weights(shape):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.05))
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    # Define the weights that will be trained using create_weights function.
    Weights = create_weights(shape=[conv_filter_size,
                                    conv_filter_size,
                                    num_input_channels,
                                    num_filters])
    # Create biases using the create_biases function. These are also trained.
    Biases = create_biases(num_filters)
    # Creating the convolutional layer.
    Layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding=’SAME’)
    layer += biases
    # Use max-pooling.
    Layer = tf.nn.max_pool2d(input=layer,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding=’SAME’)
    # Output of pooling is fed to Relu which is the activation function for us.
    Layer = tf.nn.relu(layer)
    return layer

def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Define trainable weights and biases.
    Weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Connected layer takes input x and produces wx+b.Since,
    # these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

def create_flatten_layer(layer):
    # The shape of the layer will be [batch_size img_size img_size num_channels]
    # But let’s get it from the previous layer.
    Layer_shape = layer.get_shape()

    # Number of features will be img_height * img_width* num_channels.
    # Calculate it in place of hard-coding it.
    Num_features = layer_shape[1:4].num_elements()

    # Now, flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])
    return layer

# Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128

layer_conv1 = create_convolutional_layer(input=x,
                                         num_input_channels=num_channels,
                                         conv_filter_size=filter_size_conv1,
                                         num_filters=num_filters_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                         num_input_channels=num_filters_conv1,
                                         conv_filter_size=filter_size_conv2,
                                         num_filters=num_filters_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                         num_input_channels=num_filters_conv2,
                                         conv_filter_size=filter_size_conv3,
                                         num_filters=num_filters_conv3)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size,
                            use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size,
                            num_outputs=num_classes,
                            use_relu=False)

y_pred = tf.nn.softmax(layer_fc2,name=’y_pred’)

y_pred_cls = tf.math.argmax(y_pred, axis=1)
session.run(tf.compat.v1.global_variables_initializer())

y_true, y_true_cls = create_y_true()

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                           labels=y_true)

cost = tf.reduce_mean(cross_entropy)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.compat.v1.global_variables_initializer())

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = “Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}”
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

save_path = ‘plant_disease_classification/ckpts’
if not os.path.exists(save_path):
    os.makedirs(save_path)

saver = tf.compat.v1.train.Saver()

def train(num_iteration):
    global total_iterations

    for I in range(total_iterations, total_iterations + num_iteration):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}
        session.run(optimizer, feed_dict=feed_dict_tr)

        if I % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(I / int(data.train.num_examples / batch_size))
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, os.path.join(save_path, ‘plants-disease-model’))
    total_iterations += num_iteration

train(num_iteration=3000)
