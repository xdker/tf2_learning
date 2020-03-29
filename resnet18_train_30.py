import csv
import datetime
import os

import tensorflow as tf
from tensorflow.keras import optimizers

from data_tools import get_data, get_kfold, dats_pipleline, get_train_test
from net import resnet18

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(666)
acc_headers = ['step', 'acc']
loss_headers = ['step', 'loss']


# @tf.function
def train(train_db, test_db, alpha=0.001, name=None):
    if not os.path.exists("logs/" + name):
        os.makedirs("logs/" + name)
    with open("logs/" + name + '/train_acc.csv', 'w')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(acc_headers)
    with open("logs/" + name + '/train_loss.csv', 'w')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(loss_headers)
    with open("logs/" + name + '/test_acc.csv', 'w')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(acc_headers)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/" + name + '/%s' % stamp
    writer = tf.summary.create_file_writer(log_dir)
    model = resnet18(30, alpha, writer)  # ResNet18网络
    model.build(input_shape=(None, 112, 112, 3))
    model.summary()  # 统计网络参数
    optimizer = optimizers.Adam(lr=1e-4)  # 构建优化器
    max_train_step = 0
    with writer.as_default():
        for epoch in range(50):  # 训练epoch
            for step, (x, y) in enumerate(train_db):
                with tf.GradientTape() as tape:
                    logits = model(x)
                    y_onehot = tf.one_hot(y, depth=30)
                    # 计算交叉熵
                    prob = tf.nn.softmax(logits, axis=1)
                    pred = tf.argmax(prob, axis=1)
                    y = tf.cast(y, tf.int32)
                    pred = tf.cast(pred, tf.int32)
                    correct = tf.cast(tf.equal(y, pred), dtype=tf.float32)
                    train_acc = tf.reduce_mean(correct)
                    loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                    loss = tf.reduce_mean(loss)

                # 计算梯度信息
                with open("logs/" + name + '/train_acc.csv', 'a+', newline='')as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow([max_train_step * epoch + step, float(train_acc)])
                with open("logs/" + name + '/train_loss.csv', 'a+', newline='')as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow([max_train_step * epoch + step, float(loss)])
                tf.summary.scalar("loss", float(loss), step=max_train_step * epoch + step)
                tf.summary.scalar("train_acc", train_acc, step=max_train_step * epoch + step)
                grads = tape.gradient(loss, model.trainable_variables)
                # 更新网络参数
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                if step % 50 == 0:
                    print(epoch, step, 'loss:', float(loss))
                if max_train_step < step:
                    max_train_step = step
            total_num = 0
            total_correct = 0
            for x, y in test_db:
                logits = model(x)
                prob_test = tf.nn.softmax(logits, axis=1)
                pred_test = tf.argmax(prob_test, axis=1)
                correct_test = tf.cast(tf.equal(pred_test, y), dtype=tf.int32)
                correct_test = tf.reduce_sum(correct_test)
                total_num += x.shape[0]
                total_correct += int(correct_test)
            test_acc = total_correct / total_num
            tf.summary.scalar("test_acc", test_acc, step=max_train_step * epoch + step)
            print(epoch, 'test_acc:', test_acc)
            with open("logs/" + name + '/test_acc.csv', 'a+', newline='')as f:
                f_csv = csv.writer(f)
                f_csv.writerow([max_train_step * epoch + step, float(test_acc)])
        model.save_weights("logs/" + name + '/saved_model.h5')


if __name__ == '__main__':
    import numpy as np

    train_X, train_Y = get_data("data/animals/train")
    test_X, test_Y = get_data("data/animals/val")
    train_X = list(train_X)
    train_Y = list(train_Y)
    train_X.extend(list(test_X))
    train_Y.extend(list(test_Y))
    train_X = np.asarray(train_X)
    train_Y = np.asarray(train_Y)
    train_X, test_X, train_Y, test_Y = get_train_test(train_X, train_Y)
    alpha_list = [0.001, 0.01, 0.1]
    for alpha in alpha_list:
        name = str(alpha) + "_train"
        train_db = dats_pipleline(train_X, train_Y, BATCH_SIZE=32)
        test_db = dats_pipleline(test_X, test_Y, BATCH_SIZE=32)
        train(train_db, test_db, alpha, name)
