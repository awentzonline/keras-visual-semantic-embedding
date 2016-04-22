from collections import defaultdict

import numpy as np


def encode_sentences(model, vocab_map, X, max_length=None,
        embedding_dim=1024, verbose=False, batch_size=128):
    '''Encode sentences into the joint embedding space.

    This is mostly from the original @ryankiros implementation.
    '''
    n_words = len(vocab_map)
    features = np.zeros((len(X), embedding_dim), dtype='float32')

    captions = [s.split() for s in X]
    if max_length is None:
        max_length = max((len(c) for c in captions))
    else:
        captions = [s[:max_length - 1] for s in captions]
    # quick check if a word is in the dictionary
    d = defaultdict(bool)
    for w in vocab_map.keys():
        d[w] = True

    k = max_length - 1
    if verbose:
        print k
    numbatches = len(captions) / batch_size + 1
    for minibatch in range(0, len(captions), batch_size):
        caption = captions[minibatch:minibatch + batch_size]

        seqs = []
        for i, cc in enumerate(caption):
            seqs.append([
                vocab_map[w] if d[w] > 0 and vocab_map[w] < n_words else 1 for w in cc
            ])
        x = np.zeros((k + 1, len(caption))).astype('int64')
        x_mask = np.zeros((k + 1, len(caption))).astype('float32')
        for idx, s in enumerate(seqs):
            x[:len(s), idx] = s
            x_mask[:len(s) + 1, idx] = 1.

        ff = model.predict(x.transpose(1, 0))
        for ind, c in enumerate(range(minibatch, minibatch + len(caption))):
            features[c] = ff[ind]

    return features
