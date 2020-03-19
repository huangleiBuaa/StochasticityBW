import tensorflow as tf
from archs.gan import GAN
from tensorflow.python.keras.models import Input


class AC_GAN(GAN):
    def __init__(self, ce_weight_generator=0.1, ce_weight_discriminator=1, classify_generated=False, **kwargs):
        super(AC_GAN, self).__init__(**kwargs)
        self.ce_weight_generator = ce_weight_generator
        self.ce_weight_discriminator = ce_weight_discriminator
        self.classify_generated = classify_generated
        self.additional_inputs_for_discriminator_train = [Input((1,), dtype='int32')]

    def additional_discriminator_losses(self):
        losses = []
        cls_real = self.ce_weight_discriminator * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                   (labels=tf.squeeze(self.additional_inputs_for_discriminator_train[0], axis=1),
                    logits=self.discriminator_real_output[1]))
        self.discriminator_metric_names.append('cls_real')
        losses.append(cls_real)
        if self.classify_generated:
            cls_fake = self.ce_weight_discriminator * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                   (labels=tf.squeeze(self.generator_input[1], axis=1), logits=self.discriminator_fake_output[1]))
            losses.append(cls_fake)
            self.discriminator_metric_names.append('cls_fake')
        return losses

    def additional_generator_losses(self):
        cls_real = self.ce_weight_generator * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                   (labels=tf.squeeze(self.generator_input[1], axis=1), logits=self.discriminator_fake_output[1]))
        self.generator_metric_names.append('cls')
        return [cls_real]
