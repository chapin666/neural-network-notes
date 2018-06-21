import tensorflow as tf


################## matmul ###################
v1 = tf.constant([[2, 3]])
v2 = tf.constant([[2], [3]])

product = tf.matmul(v1, v2)
print(product)

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

################## variable ###################
num = tf.Variable(0, name = "count");
new_value = tf.add(num, 10);
op = tf.assign(num, new_value);

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(num))
    for i in range(5):
        sess.run(op)
        print(sess.run(num))
        
################## placeholder ###################
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

new_value = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(new_value, feed_dict={input1: 23.0, input2: 11.0}))
