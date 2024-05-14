import tensorflow as tf


def wide_basic(inputs, in_planes, out_planes, stride):
    if stride != 1 or in_planes != out_planes:
        skip_c = tf.keras.layers.Conv2D(out_planes, kernel_size=1, strides=stride, use_bias=True, padding='SAME')(
            inputs)
    else:
        skip_c = inputs

    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(out_planes, kernel_size=3, strides=1, use_bias=True, padding='SAME')(x)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    x = tf.keras.layers.BatchNormalization(scale=True, center=True, )(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(out_planes, kernel_size=3, strides=stride, use_bias=True, padding='SAME')(x)

    # print("skip:", skip_c.shape)
    # print("x:", x.shape)
    x = tf.keras.layers.add([skip_c, x])

    return x


def wide_layer(out, in_planes, out_planes, num_blocks, stride):
    strides = [stride] + [1] * int(num_blocks - 1)
    # print("strides:", strides)
    for strid in strides:
        # print("i:", i)
        out = wide_basic(out, in_planes, out_planes, strid)
        in_planes = out_planes

    return out


def make_big_resnet_filter(ins, depth=28, widen_factor=10, model_size=100):
    n = (depth - 4) / 6
    k = widen_factor
    nStages = [16, 16 * k, 32 * k, 64 * k]
    x = tf.keras.layers.Conv2D(nStages[0], kernel_size=3, strides=1, use_bias=True, padding='SAME')(ins)
    x = wide_layer(x, nStages[0], nStages[1], n, stride=1)
    x = wide_layer(x, nStages[1], nStages[2], n, stride=2)
    x = wide_layer(x, nStages[2], nStages[3], n, stride=2)
    x = tf.keras.layers.BatchNormalization(scale=True, center=True)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.AvgPool2D([8, 8])(x)
    x = tf.reshape(x, (-1, 640))
    x = tf.keras.layers.Dense(model_size)(x)
    return x


def make_small_resnet_filter(ins, depth=28, widen_factor=10, model_size=10):
    n = (depth - 4) / 6
    k = widen_factor
    nStages = [16, 16 * k, 32 * k, 64 * k]
    x = tf.keras.layers.Conv2D(nStages[0], kernel_size=3, strides=1, use_bias=True, padding='SAME')(ins)
    x = wide_layer(x, nStages[0], nStages[1], n, stride=1)
    x = wide_layer(x, nStages[1], nStages[2], n, stride=2)
    x = wide_layer(x, nStages[2], nStages[3], 1, stride=2)
    x = tf.keras.layers.BatchNormalization(scale=True, center=True)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.AvgPool2D([8, 8])(x)
    x = tf.reshape(x, (-1, 640))
    x = tf.keras.layers.Dense(model_size)(x)
    return x



def make_big_resnet_filter_main(ins, depth=28, widen_factor=10, model_size=100):
    n = (depth - 4) / 6
    k = widen_factor
    nStages = [16, 16 * k, 32 * k, 64 * k]
    x = tf.keras.layers.Conv2D(nStages[0], kernel_size=3, strides=1, use_bias=True, padding='SAME')(ins)
    x = wide_layer(x, nStages[0], nStages[1], n, stride=1)
    x = wide_layer(x, nStages[1], nStages[2], n, stride=2)
    x = wide_layer(x, nStages[2], nStages[3], n, stride=2)
    x = tf.keras.layers.BatchNormalization(scale=True, center=True)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.AvgPool2D([8, 8])(x)
    x = tf.reshape(x, (-1, 640))
    x = tf.keras.layers.Dense(model_size)(x)
    return x


def make_small_resnet_filter_main(ins, depth=28, widen_factor=10, model_size=10):
    n = (depth - 4) / 6
    k = widen_factor
    nStages = [16, 16 * k, 32 * k, 64 * k]
    x = tf.keras.layers.Conv2D(nStages[0], kernel_size=3, strides=1, use_bias=True, padding='SAME')(ins)
    x = wide_layer(x, nStages[0], nStages[1], n, stride=1)
    x = wide_layer(x, nStages[1], nStages[2], n, stride=2)
    x = wide_layer(x, nStages[2], nStages[3], 1, stride=2)
    x = tf.keras.layers.BatchNormalization(scale=True, center=True)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.AvgPool2D([8, 8])(x)
    x = tf.reshape(x, (-1, 640))
    x = tf.keras.layers.Dense(model_size)(x)
    return x

