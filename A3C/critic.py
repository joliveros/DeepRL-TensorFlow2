import tensorflow as tf
from exchange_data.models.resnet.model import Model
from tensorflow.keras.layers import Dense

class Critic:
    def __init__(self, state_dim, critic_lr, **kwargs):
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.model = self.create_model(**kwargs)
        self.opt = tf.keras.optimizers.Adam(self.critic_lr)

    def create_model(self, **kwargs):
        model = Model(
            input_shape=self.state_dim,
            **kwargs
        )

        print(model.summary())

        dense = Dense(1, activation='linear')(model.output)

        return tf.keras.Model(
            inputs=model.inputs,
            outputs=[dense])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
