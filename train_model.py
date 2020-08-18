'''
This module describes the contrastive predictive coding model from DeepMind:

Oord, Aaron van den, Yazhe Li, and Oriol Vinyals.
"Representation Learning with Contrastive Predictive Coding."
arXiv preprint arXiv:1807.03748 (2018).
'''
from data_utils import SortedNumberGenerator
from os.path import join, basename, dirname, exists
import keras
from keras import backend as K
import tensorflow as tf


def network_encoder(x, code_size):

    ''' Define the network mapping images to embeddings '''
    # x_shape = (B, 64, 64, 3), code_size = 128
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)  # (B, 31, 31, 64)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)  # (B, 15, 15, 64)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)  # (B, 7, 7, 64)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)  # (B, 3, 3, 64)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)  # (B, 576)
    x = keras.layers.Dense(units=256, activation='linear')(x)  # (B, 256)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)  # (B, 128)

    return x


def network_autoregressive(x):

    ''' Define the network that integrates information along the sequence '''

    # x = keras.layers.GRU(units=256, return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)
    # x_shape = (B, 4, 128)
    x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)  # (B, 256)

    return x


def network_prediction(context, code_size, predict_terms):

    ''' Define the network mapping context to multiple embeddings '''
    # context_shape = (B, 256)
    outputs = []
    for i in range(predict_terms):
        outputs.append(keras.layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))
    # output_shape = (predict_term, B, 128)

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)
    # output_shape = (B, predict_term, 128)

    return output


class CPCLayer(keras.layers.Layer):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        Reference
        ----------
        - https://github.com/pat-coady/contrast-pred-code/blob/master/replearn/models.py

        """
        del kwargs  # not use

        # TODO: stop_gradient

        # Compute dot product among vectors
        preds, y_pos_encoded, y_neg_encoded = inputs  # (B, 4, 128), (B, 4, 128), (B, 4, num_neg, 128)
        preds = K.expand_dims(preds, axis=-2)  # (B, 4, 1, 128)
        y_pos_encoded = K.expand_dims(y_pos_encoded, axis=-2)  # (B, 4, 1, 128)
        pos_dot_product = K.mean(preds * y_pos_encoded, axis=-1)  # (B, 4, 1)
        neg_dot_product = K.mean(preds * y_neg_encoded, axis=-1)  # (B, 4, num_neg)
        dot_product = K.concatenate([pos_dot_product, neg_dot_product], axis=-1)  # (B, 4, num_neg + 1)
        # dot_product = K.mean(y_pos_encoded * preds, axis=-1)  # (B, 4)
        # dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension, (B, 1)

        # Keras loss functions take probabilities
        # dot_product_probs = K.sigmoid(dot_product)
        # minus the maximum dot product to ensure numerical stability
        dot_product_probs = K.softmax(dot_product - K.max(dot_product, axis=-1, keepdims=True))  # (B, 4, num_neg + 1)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return input_shape[2][0], input_shape[2][1], input_shape[2][2] + 1


def network_cpc(image_shape, terms, predict_terms, code_size, num_neg, learning_rate):
    """Define the CPC network combining encoder and autoregressive model

    References
    ----------
    - https://stackoverflow.com/questions/44627977/keras-multi-inputs-attributeerror-nonetype-object-has-no-attribute-inbound-n

    """

    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    K.set_learning_phase(1)

    # Define encoder model
    encoder_input = keras.layers.Input(image_shape)  # (B, 64, 64, 3)
    encoder_output = network_encoder(encoder_input, code_size)  # (B, 128)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()

    # Define rest of model
    x_input = keras.layers.Input((terms, image_shape[0], image_shape[1], image_shape[2]))  # (B, 4, 64, 64, 3)
    x_encoded = keras.layers.TimeDistributed(encoder_model)(x_input)  # (B, 4, 128)
    context = network_autoregressive(x_encoded)  # (B, 256)
    preds = network_prediction(context, code_size, predict_terms)  # (B, 4, 128)

    y_pos_input = keras.layers.Input((predict_terms, image_shape[0], image_shape[1], image_shape[2]))  # (B, 4, 64, 64, 3)
    y_pos_encoded = keras.layers.TimeDistributed(encoder_model)(y_pos_input)  # (B, 4, 128)

    y_neg_input = keras.layers.Input((predict_terms, num_neg, image_shape[0], image_shape[1], image_shape[2]))  # (B, 4, num_neg, 64, 64, 3)
    y_neg_encoded = []
    for idx in range(num_neg):
        y_neg_input_slice = keras.layers.Lambda(lambda x: x[:, :, idx])(y_neg_input)  # (B, 4, 64, 64, 3)
        y_neg_encoded.append(keras.layers.TimeDistributed(encoder_model)(y_neg_input_slice))  # (B, 4, 128)

    y_neg_encoded = keras.layers.Lambda(lambda x: K.stack(x, axis=-2))(y_neg_encoded)  # (B, 4, num_seg, 128)

    # Loss
    dot_product_probs = CPCLayer()([preds, y_pos_encoded, y_neg_encoded])

    # Model
    cpc_model = keras.models.Model(inputs=[x_input, y_pos_input, y_neg_input], outputs=dot_product_probs)

    # Compile model
    cpc_model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    cpc_model.summary()

    return cpc_model


def train_model(epochs, batch_size, output_dir, code_size, lr=1e-4, terms=4, predict_terms=4, num_neg_samples=7,
                image_size=28, color=False):

    # Prepare data
    train_data = SortedNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
                                       negative_samples=num_neg_samples, predict_terms=predict_terms,
                                       image_size=image_size, color=color, rescale=True)

    validation_data = SortedNumberGenerator(batch_size=batch_size, subset='valid', terms=terms,
                                            negative_samples=num_neg_samples, predict_terms=predict_terms,
                                            image_size=image_size, color=color, rescale=True)

    # Prepares the model
    model = network_cpc(image_shape=(image_size, image_size, 3), terms=terms, predict_terms=predict_terms,
                        code_size=code_size, num_neg=num_neg_samples, learning_rate=lr)

    # Callbacks
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4)]

    # Trains the model
    model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )

    # Saves the model
    # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
    model.save(join(output_dir, 'cpc.h5'))

    # Saves the encoder alone
    encoder = model.layers[1].layer
    encoder.save(join(output_dir, 'encoder.h5'))


if __name__ == "__main__":

    train_model(
        epochs=10,
        batch_size=32,
        output_dir='models/64x64',
        code_size=128,
        lr=1e-3,
        terms=4,
        predict_terms=4,
        num_neg_samples=3,
        image_size=64,
        color=True
    )

