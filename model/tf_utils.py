import tensorflow as tf

def get_cells(cell_type, hidden_units, num_layers, dropout):
    cells = []
    for _ in range(num_layers):
        if cell_type.lower() == 'lstm':
            cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units)
        else:
            cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_units)
        if dropout is not None:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
        cells.append(cell)
    return cells

def get_multi_layer_rnn(cell_type, hidden_units, num_layers, dropout=None):
    cells = get_cells(cell_type, hidden_units, num_layers, dropout)
    return tf.nn.rnn_cell.MultiRNNCell(cells)

def get_multi_layer_rnn_attn(cell_type, batch_size, hidden_units, num_layers, encoder_outputs, encoder_states,
                             encoder_lengths, dropout=None, attn='bahdanau'):
    if attn.lower() == 'bahdanau':
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention
    else:
        attention_mechanism = tf.contrib.seq2seq.LuongAttention

    attention_mechanism = attention_mechanism(num_units=hidden_units, memory=encoder_outputs,
                                              memory_sequence_length=encoder_lengths)

    decoder_cells = get_cells(cell_type, hidden_units, num_layers, dropout)

    decoder_cells[-1] = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cells[-1],
                                                            attention_mechanism=attention_mechanism,
                                                            attention_layer_size=hidden_units,
                                                            initial_cell_state=encoder_states[-1])

    decoder_states = [es for es in encoder_states]
    decoder_states[-1] = decoder_cells[-1].zero_state(batch_size=batch_size, dtype=tf.float32)
    decoder_states = tuple(decoder_states)
    return tf.nn.rnn_cell.MultiRNNCell(decoder_cells), decoder_states