from deepiv.models import Treatment, Response
import deepiv.architectures as architectures
import deepiv.densities as densities

from keras.layers import Input, Dense
from keras.layers.merge import Concatenate

class DeepIV():
    def __init__(self, num_instruments, num_features, num_treatments,
        hidden_layer_sizes = [128, 64, 32],
        num_gaussians = 10, dropout_rate = 0.2,
        l2_penalty = 0.0001, activation_function = "relu",
            optimizer = "adam"):

        instruments = Input(shape = (num_instruments,), name = "instruments")
        features = Input(shape = (num_features,), name = "features")
        treatments = Input(shape = (num_treatments,), name = "treatment")

        instruments_and_features = Concatenate(axis=1)([instruments, features])
        features_and_treatments = Concatenate(axis=1)([features, treatments])

        estimated_treatments = architectures.feed_forward_net(
        instruments_and_features,
        lambda x: densities.mixture_of_gaussian_output(x, num_gaussians),
        hidden_layers = hidden_layer_sizes,
        dropout_rate = dropout_rate,
        l2 = l2_penalty,
        activations = activation_function)

        treatment_model = Treatment(inputs=[instruments, features], outputs=estimated_treatments)
        treatment_model.compile(optimizer, loss="mixture_of_gaussians", n_components = num_gaussians)

        estimated_response = architectures.feed_forward_net(
        features_and_treatments,
        Dense(1),
        hidden_layers = hidden_layer_sizes,
        dropout_rate=dropout_rate,
        l2 = l2_penalty,
        activations = activation_function)

        response_model = Response(
        treatment = treatment_model,
        inputs = [features, treatments],
        outputs = estimated_response)
        response_model.compile(optimizer, loss = "mse")

        self.treatment_model = treatment_model
        self.response_model = response_model


    def fit(self, Z, X, T, Y, epochs = 300, batch_size = 100, verbose = True):
        self.treatment_model.fit([Z, X], T, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.response_model.fit([Z, X], Y, epochs=epochs, batch_size=batch_size,
                    samples_per_batch=2, verbose=verbose)


    def get_expected_representation(self, X, Z, n_samples=100):
        return self.response_model.expected_representation(X, Z, n_samples=n_samples)

    def get_eta_bar(self, X, Z):
        return self.get_expected_representation(X,Z)

    def get_conditional_representation(self, X, T):
        return self.response_model.conditional_representation(X, T)

    def get_eta(self, X, T):
        return self.get_conditional_representation(X,T)

    def predict(self, X, T):
        return self.response_model.predict([X, T])
