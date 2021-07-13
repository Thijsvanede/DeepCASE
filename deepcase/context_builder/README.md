# Context Builder
The Context Builder provides an interface for its underlying attention-based encoder-decoder neural network.

## Components
The Context Builder contains three linked networks:
 * `Encoder`, encodes input events into a `context_vector` and embedded representation of the input `X'`.
 * `DecoderAttention`, decodes the `context_vector` and any previous output into an `attention` vector.
 * `DecoderEvent`, uses the `attention` vector and encoded inputs `X'` to predict the subsequent event `y`.

## API
The ContextBuilder provides the following methods:
 * `fit(X, y)`, to train the model.
 * `predict(X)`, to predict future events and compute attention.
 * `query(X, y)`, to find the optimal query given the observed event `y`.
