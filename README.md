This implementation includes a seq2seq and a variational autoencoder made of RNN.
It is furthermore implemented an attention layer along with a variational attention layer.

The feedforward is made of the following steps:
* Encoder computes feedforward of the input graph
* The hidden state of the last layer is saved
* If the architecture is a vae
    * Two linear layers compute mu and logvar and the reparameterization trick is performed
    * The latent dimension is transformed into the decoder hidden state size via a linear layer
* The decoder hidden state is initialized to the computed vector (Either directly from the encoder or from the latent space)
* The generator generates a graph given the initial state.

# Roadmap

* args.py contains the parameters required to train the model
* conditioning_comparison.py evaluates the MMD of one or multiple generated graph wrt the original graph fed as input
* evaluate_cond.py evaluates the MMD of a set of generated graphs
* main_cond.py trains the full autoencoder