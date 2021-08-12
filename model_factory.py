from models.MelGAN import GeneratorMel, GeneratorMelMix, DiscriminatorMel, SpecDiscriminator
from models.Demucs import Demucs

class ModelFactory():
    def __init__(self, config):
        self.config = config
    def generator(self):
        if self.config.model == "melgan":
            if self.config.use_mix:
                return GeneratorMelMix(self.config.n_mel_channels, self.config.ngf, self.config.n_residual_layers,self.config.skip_cxn)
            else:
                return GeneratorMel(self.config.n_mel_channels, self.config.ngf, self.config.n_residual_layers,self.config.skip_cxn)
        if self.model == "demucs":
            return Demucs([self.config.source],audio_channels=self.config.audio_channels,  segment_length=int(self.config.segment_duration * self.config.sample_rate), skip_cxn = self.config.skip_cxn)
        else:
            raise ValueError('Invalid Model')
    def discriminator(self):
        if self.multi_disc:
            return DiscriminatorMel(self.config.num_D, self.config.ndf, self.config.n_layers_D, self.config.downsamp_factor), SpecDiscriminator(self.config.n_mel_channels)
        else:
            return DiscriminatorMel(self.config.num_D, self.config.ndf, self.config.n_layers_D, self.config.downsamp_factor)
