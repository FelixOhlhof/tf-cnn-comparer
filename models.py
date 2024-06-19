import tensorflow as tf

def tensorflow_flowers_demo(img_height, img_width):
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip("horizontal_and_vertical"),
      tf.keras.layers.RandomRotation(0.2),
    ])

    model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1) # binary classifier
    ])
    #compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model

def demo2(img_height, img_width):
    #create the model
    data_augmentation = tf.keras.Sequential(
      [
        tf.keras.layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
      ]
    )

    model = tf.keras.Sequential([
      tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    #compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

    return model

def mobile_net_v2(img_height, img_width):
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip('horizontal'),
      tf.keras.layers.RandomRotation(0.2),
    ])
    
    rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                               include_top=False,
                                               weights='imagenet')
    
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = rescale(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0010
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

    return model


def alexnet(img_height, img_width):
    data_augmentation = tf.keras.Sequential(
    [
              tf.keras.layers.RandomFlip("horizontal",
                                input_shape=(img_height,
                                            img_width,
                                            3)),
              tf.keras.layers.RandomRotation(0.1),
              tf.keras.layers.RandomZoom(0.1),
    ]
    )

    model = tf.keras.models.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu',
                               input_shape=(img_height, img_width, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.name = 'AlexNet'
    model.compile( loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model


def xception(img_height, img_width):

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])

    rescale = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)
    base_model = tf.keras.applications.Xception(input_shape=(img_height, img_width, 3),
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = rescale(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.00005
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

    return model


def convnextxlarge(img_height, img_width):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])

    rescale = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)
    base_model = tf.keras.applications.ConvNeXtXLarge(input_shape=(img_height, img_width, 3),
                                                include_top=False,
                                                weights='imagenet')

    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = rescale(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.00001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

    return model


def inceptionv3(img_height, img_width):

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])

    rescale = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)
    base_model = tf.keras.applications.InceptionV3(input_shape=(img_height, img_width, 3),
                                                include_top=False,
                                                weights='imagenet')

    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = rescale(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.00005
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

    return model


def single_conv2d_with_aug(img_height, img_width):
    data_augmentation = tf.keras.Sequential(
    [
              tf.keras.layers.RandomFlip("horizontal",
                                input_shape=(img_height,
                                            img_width,
                                            3)),
              tf.keras.layers.RandomRotation(0.1),
              tf.keras.layers.RandomZoom(0.1),
    ]
    )

    model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.name = 'single_conv2d_with_aug'

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

    return model

