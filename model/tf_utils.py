import tensorflow as tf

def get_multi_layer_rnn(cell_type, hidden_units, num_layers, dropout=None):
    cells = []
    for _ in range(num_layers):
        if cell_type.lower() == 'lstm':
            cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units)
        else:
            cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_units)
        if dropout is not None:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
        cells.append(cell)

    return tf.nn.rnn_cell.MultiRNNCell(cells)