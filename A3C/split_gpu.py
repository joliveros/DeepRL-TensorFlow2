import tensorflow as tf


def split_gpu(memory=2400):
        physical_devices = tf.config.list_physical_devices('GPU')

        if len(physical_devices) > 0 and memory > 0:
            tf.config.set_logical_device_configuration(
                physical_devices[0],
                [
                    tf.config.
                    LogicalDeviceConfiguration(memory_limit=memory),
                ])
