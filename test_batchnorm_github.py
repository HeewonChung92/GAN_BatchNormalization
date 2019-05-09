import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as sio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### Basic GAN
total_epochs = 300
batch_size = 64
learningrate_dis = 0.0002
learningrate_gen = 0.0002

num_input = 640
num_noise = 256
num_hidden = 448
num_output = 1

myinit = tf.random_normal_initializer(mean=0.0, stddev=0.01)
# generator
def generator(z, train_state):
    with tf.variable_scope(name_or_scope="Gen") as scope:
        gw1 = tf.get_variable(name="w1", shape=[num_noise, num_hidden], initializer=myinit)
        gb1 = tf.get_variable(name="b1", shape=[num_hidden], initializer=myinit)
        gw2 = tf.get_variable(name="w2", shape=[num_hidden, num_input], initializer=myinit)
        gb2 = tf.get_variable(name="b2", shape=[num_input], initializer=myinit)

    fcHidden = tf.matmul(z, gw1) + gb1
    bnHidden = tf.layers.batch_normalization(fcHidden, training=train_state)
    # hidden = tf.nn.leaky_relu(bnHidden)
    hidden = tf.nn.relu(bnHidden)
    logits = tf.matmul(hidden, gw2) + gb2
    bnLogits = tf.layers.batch_normalization(logits, training=train_state)
    output = tf.nn.sigmoid(bnLogits)
    return output, logits, hidden, tf.nn.leaky_relu(fcHidden)

# Pre-discriminator
def discriminator_pre(x, train_state = True):
    with tf.variable_scope(name_or_scope="Pre") as scope:
        dw1 = tf.get_variable(name="w1", shape=[num_input, num_hidden], initializer=myinit)
        db1 = tf.get_variable(name="b1", shape=[num_hidden], initializer=myinit)
        dw2 = tf.get_variable(name="w2", shape=[num_hidden, num_output], initializer=myinit)
        db2 = tf.get_variable(name="b2", shape=[num_output], initializer=myinit)

    fcHidden = tf.matmul(x, dw1) + db1
    bnHidden = tf.layers.batch_normalization(fcHidden, training=train_state)
    # hidden = tf.nn.leaky_relu(bnHidden)
    hidden = tf.nn.relu(bnHidden)
    logits = tf.matmul(hidden, dw2) + db2
    bnLogits = tf.layers.batch_normalization(logits, training=train_state)
    output = tf.nn.sigmoid(bnLogits)
    return output, logits

# discriminator
def discriminator(x , train_state, reuse = False):
    with tf.variable_scope(name_or_scope="Dis", reuse=reuse) as scope:
        dw1 = tf.get_variable(name="w1", shape=[num_input, num_hidden], initializer=myinit)
        db1 = tf.get_variable(name="b1", shape=[num_hidden], initializer=myinit)
        dw2 = tf.get_variable(name="w2", shape=[num_hidden, num_output], initializer=myinit)
        db2 = tf.get_variable(name="b2", shape=[num_output], initializer=myinit)

    fcHidden = tf.matmul(x, dw1) + db1
    bnHidden = tf.layers.batch_normalization(fcHidden, training=train_state)
    # hidden = tf.nn.leaky_relu(bnHidden)
    hidden = tf.nn.relu(bnHidden)
    logits = tf.matmul(hidden, dw2) + db2
    bnLogits = tf.layers.batch_normalization(logits, training=train_state)
    output = tf.nn.sigmoid(bnLogits)
    return output, logits

def random_noise(batch_size):
    return np.random.normal(size=[batch_size, num_noise])

def random_ones(batch_size):
    return np.ones(shape=[batch_size, 1])

g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.float32, [None, num_input]) # GAN 은 autoencoder 와 마찬가지로 unsupervised learning 이므로 y(label)을 사용하지 않습니다.
    Z = tf.placeholder(tf.float32, [None, num_noise]) # Z 는 생성기의 입력값으로 사용될 노이즈 입니다.
    preLabel = tf.placeholder(tf.float32, [None, 1])
    trainingState = tf.placeholder(tf.bool)

    # Pre-train
    result_of_pre, logits_pre = discriminator_pre(X)
    p_loss = tf.reduce_mean(tf.square(result_of_pre - preLabel))

    # Discriminator & Generator
    fake_x, fake_logits, ghidden, gohidden  = generator(Z, trainingState)
    result_of_real, logits_real = discriminator(X, trainingState)
    result_of_fake, logits_fake = discriminator(fake_x, trainingState, True)

    # Discriminator / Generator 손실 함수를 정의합니다.
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(result_of_real)))  # log(D(x))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(result_of_fake)))  # log(1-D(G(z)))
    d_loss = d_loss_real + d_loss_fake  # log(D(x)) + log(1-D(G(z)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(result_of_fake)))  # log(D(G(z)))

    # Parameter
    t_vars = tf.trainable_variables() # return list
    g_vars = [var for var in t_vars if "Gen" in var.name]
    d_vars = [var for var in t_vars if "Dis" in var.name]
    p_vars = [var for var in t_vars if "Pre" in var.name]

    # Optimizer / Gradient
    p_train = tf.train.AdamOptimizer(learning_rate=learningrate_dis, beta1=0.5, beta2=0.999).minimize(p_loss, var_list=p_vars)
    g_train = tf.train.AdamOptimizer(learning_rate=learningrate_gen, beta1=0.5, beta2=0.999).minimize(g_loss, var_list=g_vars)
    d_train = tf.train.AdamOptimizer(learning_rate=learningrate_dis, beta1=0.5, beta2=0.999).minimize(d_loss, var_list=d_vars)

    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     g_train = tf.train.AdamOptimizer(learning_rate=learningrate_gen, beta1=0.5, beta2=0.999).minimize(g_loss, var_list=g_vars)
    #     d_train = tf.train.AdamOptimizer(learning_rate=learningrate_dis, beta1=0.5, beta2=0.999).minimize(d_loss, var_list=d_vars)

    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # p_train = tf.train.AdamOptimizer(learning_rate=learningrate_dis, beta1=0.5, beta2=0.999).minimize(p_loss, var_list=p_vars)
    # p_train = tf.group([p_train, update_ops])


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    contentsMat = sio.loadmat('data.mat')
    dataPPG = contentsMat['data']

    trainSize = dataPPG.shape[0]
    total_batchs = int(trainSize / batch_size)
    print('Total :   ', total_batchs, trainSize)

    SaverDir = './zTest_batchNorm_ReLU/'
    writer = tf.summary.FileWriter(SaverDir, sess.graph)

    if not os.path.exists(SaverDir):
        os.makedirs(SaverDir)

    ###========== Pre-Train
    arrayPre = []
    for epoch in range(50):
        for batch in range(total_batchs):
            batch_x = dataPPG[batch * batch_size: (batch + 1) * batch_size]
            batch_label = random_ones(batch_size)

            _, pl = sess.run([p_train, p_loss], feed_dict={X: batch_x, preLabel: batch_label})
            arrayPre.append([pl])
            print('Pre-Train Loss : ', epoch, ' => ', pl)
    print('Finished Pre-Train')

    weightsPre_t = sess.run(p_vars) # store pre-trained variables
    # copy weights from pre-training over to new D network
    tf.global_variables_initializer().run()
    for i, v in enumerate(d_vars):
        sess.run(v.assign(weightsPre_t[i]))

    weightsDis_t = sess.run(d_vars)
    sio.savemat(SaverDir + 'aPre.mat', {'pre_t': weightsPre_t, 'dis_t': weightsDis_t})

    ###========== Post-Train
    arrayDsc = []
    arrayGen = []
    arrayReal = []
    arrayFake = []
    summaryNumber = 0
    for epoch in range(total_epochs):
        for batch in range(total_batchs):
            batch_x = dataPPG[batch * batch_size: (batch + 1) * batch_size]
            batch_noise = random_noise(batch_size)

            _, dl, dreal, dfake = sess.run([d_train, d_loss, d_loss_real, d_loss_fake], feed_dict={X: batch_x, Z: batch_noise, trainingState: True})
            _, gl, fx = sess.run([g_train, g_loss, fake_x], feed_dict={Z: batch_noise, trainingState: True})
            gntLogits, gntHidden, realOutput, realLogit, fakeOutput, fakeLogit, gntHidden2 = sess.run([fake_logits, ghidden, result_of_real, logits_real, result_of_fake, logits_fake, gohidden], feed_dict={X: batch_x, Z: batch_noise, trainingState: True})

            arrayDsc.append([dl])
            arrayGen.append([gl])
            arrayReal.append([dreal])
            arrayFake.append([dfake])

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Discriminator", simple_value=float(dl))]), summaryNumber)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Generator", simple_value=float(gl))]), summaryNumber)
            summaryNumber += 1

            print("======= Epoch : ", epoch, " batch : ", batch, " =======================================")
            print("생성기(Generator) 성능 : ", gl)
            print("분류기(Discriminator) 성능 : ", dl, dreal, dfake)
            print("생성기와 분류기 선의의 경쟁중...")

        if (epoch == 0) or ((epoch+1) % 10 == 0):
            sample_noise = random_noise(4)
            generated = sess.run(fake_x, feed_dict={Z: sample_noise, trainingState: False})
            fig, ax = plt.subplots(5, 1)
            for i in range(4):
                ax[i].set_axis_off()
                ax[i].plot(generated[i])
            ax[4].set_axis_off()
            ax[4].plot(fx[0])
            plt.savefig(SaverDir + '{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

            weightsPre, weightsDis, weightsGen = sess.run([p_vars, d_vars, g_vars])
            sio.savemat(SaverDir + 'Epoch_' + str(epoch) + '.mat', {'pre': weightsPre, 'post': weightsDis, 'gen': weightsGen, 'batchX': batch_x, 'batchNoise': batch_noise, 'fakeX': fx,
                                                                    'generated': generated, 'sample_noise': sample_noise, 'gntLogit': gntLogits, 'gntHidden': gntHidden, 'gntHiddenOrg': gntHidden2,
                                                                    'realOutput': realOutput, 'realLogit': realLogit, 'fakeOutput': fakeOutput, 'fakeLogit': fakeLogit})

    sio.savemat(SaverDir + 'aLoss.mat', {'pre': arrayPre, 'dis': arrayDsc, 'gen': arrayGen, 'real': arrayReal, 'fake': arrayFake})
    print('최적화 완료!')