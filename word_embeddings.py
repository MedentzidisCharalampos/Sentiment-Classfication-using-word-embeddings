#Setup

import io
import os
import re
import shutil
import string
import tensorflow as tf

from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

#Download the IMDb Dataset

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)

#The /train directory has pos and neg folders with movie reviews labelled as positive and negative respectively.

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

#The train directory also has additional folders which should be removed before creating training dataset.

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

#Create a tf.data.Dataset using tf.keras.preprocessing.text_dataset_from_directory
#Use the train directory to create both train and validation datasets with a split of 20% for validation.

batch_size = 1024
seed = 123
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)

#Take a look at a few movie reviews and their labels (1: positive, 0: negative) from the train dataset.

for text_batch, label_batch in train_ds.take(1):
  for i in range(5):
    print(label_batch[i].numpy(), text_batch.numpy()[i])

#Configure the dataset for performance
#.cache() keeps data in memory after it's loaded off disk. This will ensure the dataset does not become a bottleneck while training your model.
# If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache, which is more efficient to read than many small files.

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Using the Embedding layer
#The Embedding layer can be understood as a lookup table that maps from integer indices (which stand for specific words) to dense vectors (their embeddings).
# The dimensionality (or width) of the embedding is a parameter you can experiment with to see what works well for your problem.

# Embed a 1,000 word vocabulary into 5 dimensions.
embedding_layer = tf.keras.layers.Embedding(1000, 5)

#Once trained, the learned word embeddings will roughly encode similarities between words.
#If you pass an integer to an embedding layer, the result replaces each integer with the vector from the embedding table.

result = embedding_layer(tf.constant([1,2,3]))
result.numpy()

#For text or sequence problems, the Embedding layer takes a 2D tensor of integers, of shape (samples, sequence_length), where each entry is a sequence of integers.
# It can embed sequences of variable lengths.
# You could feed into the embedding layer above batches with shapes (32, 10) (batch of 32 sequences of length 10) or (64, 15) (batch of 64 sequences of length 15).
#The returned tensor has one more axis than the input, the embedding vectors are aligned along the new last axis.
# Pass it a (2, 3) input batch and the output is (2, 3, N)

result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))

#When given a batch of sequences as input, an embedding layer returns a 3D floating point tensor, of shape (samples, sequence_length, embedding_dimensionality).
# To convert from this sequence of variable length to a fixed representation there are a variety of standard approaches.
# You could use an RNN, Attention, or pooling layer before passing it to a Dense layer.

#Text preprocessing
#Next, define the dataset preprocessing steps required for your sentiment classification model.
# Initialize a TextVectorization layer with the desired parameters to vectorize movie reviews.
# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')

# Vocabulary size and number of words in a sequence.
vocab_size = 10000
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

#Create a classification model

#Use the Keras Sequential API to define the sentiment classification model. In this case it is a "Continuous bag of words" style model.

#The TextVectorization layer transforms strings into vocabulary indices. You have already initialized vectorize_layer as a TextVectorization layer and built it's vocabulary by calling adapt on text_ds.

#Now vectorize_layer can be used as the first layer of your end-to-end classification model, feeding tranformed strings into the Embedding layer.

#The Embedding layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index.
# These vectors are learned as the model trains.
# The vectors add a dimension to the output array.
# The resulting dimensions are: (batch, sequence, embedding).

#The GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension.
# This allows the model to handle input of variable length, in the simplest way possible.

#The fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.

#The last layer is densely connected with a single output node.

embedding_dim=16

model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(),
  Dense(16, activation='relu'),
  Dense(1)
])

#Compile and train the model

#We use TensorBoard to visualize metrics including loss and accuracy. Create a tf.keras.callbacks.TensorBoard.

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

#Compile and train the model using the Adam optimizer and BinaryCrossentropy loss.

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[tensorboard_callback])

#With this approach the model reaches a validation accuracy of around 84% (note that the model is overfitting since training accuracy is higher).

#Retrieve the trained word embeddings and save them to disk
#Next, retrieve the word embeddings learned during training. The embeddings are weights of the Embedding layer in the model.
#The weights matrix is of shape (vocab_size, embedding_dimension).

vocab = vectorize_layer.get_vocabulary()
print(vocab[:10])
# Get weights matrix of layer named 'embedding'
weights = model.get_layer('embedding').get_weights()[0]
print(weights.shape)

#Write the weights to disk.
# To use the Embedding Projector, you will upload two files in tab separated format: a file of vectors (containing the embedding), and a file of meta data (containing the words).

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(vocab):
    if num == 0: continue  # skip padding token from vocab
    vec = weights[num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()

try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download('vecs.tsv')
    files.download('meta.tsv')

#Visualize the embeddings

#To visualize the embeddings, upload them to the embedding projector.

#Open the Embedding Projector (this can also run in a local TensorBoard instance).

#Click on "Load data".

#Upload the two files you created above: vecs.tsv and meta.tsv.

#The embeddings you have trained will now be displayed.
# You can search for words to find their closest neighbors.
# For example, try searching for "beautiful".
# You may see neighbors like "wonderful".








