import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

from model_function import preprocess, accuracy, test_batch_size, optimizer, INIT_POSITION

DATA_POSITION = INIT_POSITION + '/data/'



def mapp(first_y, second_y, first_dim, second_dim, attri='100-20'):
    # 获得标签的映射，这里因为有csv所以不需要
    input = tf.keras.Input([first_dim])
    output = tf.keras.layers.Dense(second_dim, use_bias=False)(input)
    model = tf.keras.Model(input, output)
    fine_labels = tf.keras.utils.to_categorical(first_y, first_dim)
    coarse_labels = tf.keras.utils.to_categorical(second_y, second_dim)
    train_db = tf.data.Dataset.from_tensor_slices((fine_labels, coarse_labels))
    train_db = train_db.shuffle(10000).map(preprocess, num_parallel_calls=2).batch(128)
    train_db = train_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    optimizer.lr.assign(0.01)
    for i in tqdm(range(10), desc="Create map csv epoch:"):
        sum_accuracy = 0
        for (x, y) in tqdm(train_db):
            with tf.GradientTape() as tape:
                pre = model(x)
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, pre))
                correct_num = tf.equal(tf.argmax(y, -1), tf.argmax(pre, -1))
                accuracy = tf.reduce_sum(tf.cast(correct_num, dtype=tf.float32))
                sum_accuracy += accuracy
                gradients = tape.gradient(cross_entropy, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(sum_accuracy / int(len(fine_labels)))
    
    dense_layer_name = 'dense_2'
    
    dense_layer_vars = np.array(model.get_layer(dense_layer_name).get_weights())
    # dense_layer_vars = model.get_layer(dense_layer_name).get_weights()
    del model
    data1 = pd.DataFrame(np.squeeze(dense_layer_vars))
    data1.to_csv(DATA_POSITION + '/' + attri + '.csv', mode='a', header=False, index=False)
    print("Mapping .csv completed")