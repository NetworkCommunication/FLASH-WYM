import tensorflow as tf
import config

z_dim = 10
h_dim = config.h_dim
image_size = config.image_size

class ANNVAE(tf.keras.Model):
    def __init__(self):
        super(ANNVAE, self).__init__()
        self.fc1 = tf.keras.layers.Dense(h_dim)
        self.fc2 = tf.keras.layers.Dense(z_dim)
        self.fc3 = tf.keras.layers.Dense(z_dim)
        self.fc4 = tf.keras.layers.Dense(h_dim)
        self.fc5 = tf.keras.layers.Dense(image_size)

    def encode(self, x):
        h = tf.nn.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = tf.exp(log_var * 0.5)
        eps = tf.random_normal(std.shape)
        return mu + eps * std

    def decode_logits(self, z):
        h = tf.nn.relu(self.fc4(z))
        return self.fc5(h)

    def decode(self, z):
        return tf.nn.sigmoid(self.decode_logits(z))

    def call(self, inputs, training=None, mask=None):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        x_reconstructed_logits = self.decode_logits(z)
        return x_reconstructed_logits, mu, log_var

"""
def vae_loss(images, x_reconstruction_logits, z_mu, z_log_sigma_sq):
    recon_loss = image_size * tf.nn.sigmoid_cross_entropy_with_logits(labels=images, logits=x_reconstruction_logits)
    recon_loss = tf.reduce_mean(recon_loss, axis=-1)

    latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq), axis=-1)

    total_loss = tf.reduce_mean(recon_loss + latent_loss)

    return total_loss
"""

def vae_loss(sequence, x_hat, z_mu, z_log_sigma_sq, beta=1.0):
    epsilon = 1e-10

    recon_loss = tf.losses.mean_squared_error(labels=sequence, predictions=x_hat)

    #recon_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=sequence, logits=x_hat)

    recon_loss = tf.reduce_mean(recon_loss)

    latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq))

    total_loss = tf.reduce_mean(recon_loss + beta * latent_loss)

    return total_loss, recon_loss, latent_loss


def train(dataset, epochs, vae_optimizer, vae_model):
    print("start epoch training")
    for epoch in range(epochs):
        for images in dataset:
            with tf.GradientTape() as vae_tape:
                x_hat, z_mu, z_log_sigma_sq = vae_model(images, training=True)

                vae_loss_value = vae_loss(images, x_hat, z_mu, z_log_sigma_sq)

            gradients_vae = vae_tape.gradient(vae_loss_value, vae_model.variables)
            print("gradient is",gradients_vae[0].numpy())
            vae_optimizer.apply_gradients(zip(gradients_vae, vae_model.variables))

        if epoch % 1 == 0:
           pass

