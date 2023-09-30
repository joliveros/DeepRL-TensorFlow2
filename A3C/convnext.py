import alog
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D


def model(input_shape, **kwargs):
    inputs = Input(shape=input_shape)

    x = tf.keras.applications.inception_resnet_v2 \
        .preprocess_input(inputs)

    model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
        include_top=False,
        # include_preprocessing=False,
        weights=None,
        # input_tensor=x,
        # input_shape=input_shape,
        pooling=None,
        classes=2,
        classifier_activation='softmax'
    )

    for layer in model.layers:
        layer.trainable = True

    x = model(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    predictions = Dense(2, activation='softmax')(x)

    return Model(inputs=inputs, outputs=predictions)
