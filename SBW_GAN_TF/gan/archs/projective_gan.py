from archs.gan import GAN
import tensorflow as tf
import tensorflow.keras.backend as K


class ProjectiveGAN(GAN):
    def __init__(self, **kwargs):
        if 'shred_disc_batch' in kwargs:
            self.shred_disc_batch = kwargs['shred_disc_batch']
        else:
            self.shred_disc_batch = False
        super(ProjectiveGAN, self).__init__(**kwargs)

    def compile_intermediate_variables(self):
        self.generator_output = [self.generator(self.generator_input), tf.identity(self.generator_input[1])]
        def shred_disc(inp):
            bs = K.shape(inp[0])[0]
            fp = [inp[0][:(bs/2)], inp[1][:(bs/2)]]
            lp = [inp[0][(bs/2):], inp[1][(bs/2):]]
            fp = self.discriminator(fp)
            lp = self.discriminator(lp)
            return K.concatenate([fp, lp], axis=0)
        if self.shred_disc_batch:
            self.discriminator_fake_output = shred_disc(self.generator_output)
            self.discriminator_real_output = shred_disc(self.discriminator_input)
        else:
            self.discriminator_fake_output = self.discriminator(self.generator_output)
            self.discriminator_real_output = self.discriminator(self.discriminator_input)
 
