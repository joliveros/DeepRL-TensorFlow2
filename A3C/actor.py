import alog
import tensorflow as tf
from exchange_data.models.resnet.model import Model


class Actor:
    def __init__(self, state_dim, action_dim, actor_lr, **kwargs):
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(self.actor_lr)
        self.entropy_beta = 0.01

    def create_model(self):
        # split_gpu()
        # return tf.keras.Sequential([
        #     Input(self.state_dim),
        #     Dense(32, activation='relu'),
        #     Dense(16, activation='relu'),
        #     Dense(self.action_dim, activation='softmax')
        # ])
        alog.info(self.state_dim)

        model = Model(
            input_shape=self.state_dim,
        )

        print(model.summary())
        return model

    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(
            actions, logits, sample_weight=tf.stop_gradient(advantages))
        entropy = entropy_loss(logits, logits)
        return policy_loss - self.entropy_beta * entropy

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(
                actions, logits, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
