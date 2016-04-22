import numpy as np
import six
from keras import backend as K
from keras.layers import Convolution2D, Dense, Embedding, GRU, Input
from keras.models import Model, Sequential

from .layers import L2Normalize
from .tools import encode_sentences


def build_image_encoder(weights=None, input_dim=4096, embedding_dim=1024, normalize=True):
    input = Input(shape=(input_dim,))
    x = Dense(
        embedding_dim,
        weights=weights
    )(input)
    if normalize:
        x = L2Normalize()(x)
    model = Model(input=input, output=x)
    return model


def build_sentence_encoder(embedding_weights=None, gru_weights=None, input_length=None, vocab_dim=32198,
        vocab_embedding_dim=300, embedding_dim=1024, normalize=True):
    # NOTE: This gives slightly different results than the original model.
    # I think it's because the original has a different masking scheme.
    model = Sequential([
        Embedding(
            vocab_dim, vocab_embedding_dim, input_length=input_length,
            weights=embedding_weights, mask_zero=True  # TODO: masking isn't quite right
        ),
        GRU(embedding_dim, weights=gru_weights, inner_activation='sigmoid'),
    ])
    if normalize:
        model.add(L2Normalize())
    return model


def build_pretrained_models(model_filename, input_length=None, normalize=True):
    img_enc_weights, embedding_weights, gru_weights, vocab_map = load_pretrained_parameters(model_filename)
    image_encoder = build_image_encoder(weights=img_enc_weights, normalize=normalize)
    sentence_encoder = build_sentence_encoder(
        embedding_weights=embedding_weights,
        gru_weights=gru_weights,
        input_length=input_length, vocab_dim=len(vocab_map),
        normalize=normalize)
    return image_encoder, sentence_encoder, vocab_map


def load_pretrained_parameters(filename):
    '''Load up the pre-trained weights from the @ryankiros implementation.
    '''
    params = np.load(filename)
    vocab_map = np.load('{}.dictionary.pkl'.format(filename))
    # image encoder weights
    if params:
        img_enc_weights = [params['ff_image_W'], params['ff_image_b']]
    else:
        img_enc_weights = None
    # sentence encoder weights
    embedding_weights = [params['Wemb']]
    W_h = params['encoder_Wx']
    U_h = params['encoder_Ux']
    b_h = params['encoder_bx']
    W_r, W_z = np.split(params['encoder_W'], 2, axis=1)
    U_r, U_z = np.split(params['encoder_U'], 2, axis=1)
    b_r, b_z = np.split(params['encoder_b'], 2)
    gru_weights = [
        W_z, U_z, b_z,
        W_r, U_r, b_r,
        W_h, U_h, b_h,
    ]
    return img_enc_weights, embedding_weights, gru_weights, vocab_map


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Visual semantic embeddings')
    parser.add_argument('model_file', type=six.text_type)
    parser.add_argument('--length', type=int, default=None)
    args = parser.parse_args()

    image_encoder, sentence_encoder, vocab_map = \
        build_pretrained_models(args.model_file, input_length=args.length)
    for enc in image_encoder, sentence_encoder:
        enc.compile(optimizer='adam', loss='mse')
    print(image_encoder.predict(np.random.uniform(0, 500, (2, 4096,))))
    print(encode_sentences(
        sentence_encoder, vocab_map,
        ['contains fancy gold', 'contains fancy gold'],
        max_length=args.length))
