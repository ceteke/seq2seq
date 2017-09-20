# Seq2Seq in Tensorflow
To run this code you have to have dependencies installed in ``requirements.txt``.  

Prerequisites:
- A Python 3.6 environment

To install dependencies to your environment simply run 
 
``source /path/to/your/environment/bin/activate``  
``pip install -r requirements.txt``  

If you have GPU, please see Tensorflow GPU installation.
#### How to use?
```python
import tensorflow as tf
from seq2seq import Seq2Seq
 
sess = tf.Session()
s2s = Seq2Seq(sess=sess,hidden_units=1024, vocab_size=20, num_layers=1, embedding_size=200)
```

### TODO
- Attention
- Different embedding matrices for encoder and decoder
- Bidirectional encoder
- Beam search