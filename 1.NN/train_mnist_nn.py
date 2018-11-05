from tensorflow.examples.tutorials.mnist import input_data
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#read MNIST dataset
mnist = input_data.read_data_sets("mnist/", one_hot = True)
train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels

# training parameters
learning_rate = 0.5
training_steps = 1000
batch_size = 100

feature_size = train_img.shape[1]
label_size = train_label.shape[1]

logs_path = 'tensorboard/'

#variables of the neural network
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, feature_size], name = 'input_data')

with tf.name_scope('labels'):
    y = tf.placeholder(tf.float32, [None, label_size], name = 'label_data')

with tf.name_scope('model-parameters'):
    w = tf.Variable(tf.zeros([feature_size, label_size]), name = 'weights')
    b = tf.Variable(tf.zeros([label_size]), name = 'bias')

#softmax
with tf.name_scope('model'):
    prediction = tf.nn.softmax(tf.matmul(x, w) + b)

#cross entrophy
with tf.name_scope('cross-entropy'):
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices = 1))
    tf.summary.scalar("loss", loss)

#gradient descent
with tf.name_scope('gradient-descent'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#accuracy
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', acc)

#initiate the training session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#save the log for visualization (tensorboard)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

#training loop
for step in range(training_steps):
    #feed the new batch and run the optimizer
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})

    #print every 50 epoch
    if step % 50 == 0:
        print("loss =", sess.run(loss, feed_dict = {x: batch_xs, y: batch_ys}))
        summary = sess.run(merged, feed_dict = {x: batch_xs, y: batch_ys})
        writer.add_summary(summary, step)

print("accuracy:", sess.run(acc, feed_dict={x: test_img, y: test_label}))

###########################################
# randomly pick an image and identifiy it #
###########################################

image_index = random.randint(0, test_img.shape[0]-1)
ans = tf.argmax(prediction, 1)
print("---");
print("identification result:", sess.run(ans, feed_dict={x: mnist.test.images[image_index:image_index+1]}))
print("label of the image: [%d]" %np.argmax(test_label[image_index, :]))

#plot the image
img = np.reshape(test_img[image_index, :], (28, 28))
plt.matshow(img, cmap = plt.get_cmap('gray'))
plt.show()

sess.close()
