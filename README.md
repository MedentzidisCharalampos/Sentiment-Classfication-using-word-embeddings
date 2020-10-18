# Sentiment-Classfication-using-Word-Embeddings

# Word Embeddings

Word embeddings give us a way to use an efficient, dense representation in which similar words have a similar encoding. An embedding is a dense vector of floating point values. Instead of specifying the values for the embedding manually, they are trainable parameters (weights learned by the model during training, in the same way a model learns weights for a dense layer). It is common to see word embeddings that are 8-dimensional (for small datasets), up to 1024-dimensions when working with large datasets. A higher dimensional embedding can capture fine-grained relationships between words, but takes more data to learn.


# The Dataset  
We use the  Large Movie Review Dataset more can be found there: http://ai.stanford.edu/~amaas/data/sentiment/. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. 

# Text Preprocessing

We create a Vectorization Layer to:
1. Strip HTML break tags
2. Split the sentences into words
3. Map words to integers

# Create a Classification Model

Use the Keras Sequential API to define the sentiment classification model. In this case it is a "Continuous bag of words" style model.

1. The TextVectorization layer transforms strings into vocabulary indices. The vectorize_layer can be used as the first layer of your end-to-end classification model, feeding tranformed strings into the Embedding layer.

2. The Embedding layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index. These vectors are learned as the model trains. The vectors add a dimension to the output array. The resulting dimensions are: (batch, sequence, embedding). The embedded dimesion is 16.

3. The GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input of variable length, in the simplest way possible.

4. The fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.

5. The last layer is densely connected with a single output node.

Compile and train the model using the `Adam` optimizer and `BinaryCrossentropy` loss. 

# Visualize the embeddings

To visualize the embeddings, upload them to the embedding projector (http://projector.tensorflow.org/).

Click on "Load data".

Upload the two files you created above: vecs.tsv and meta.tsv.

The embeddings you have trained will now be displayed. You can search for words to find their closest neighbors. For example, try searching for "beautiful". You may see neighbors like "wonderful".


