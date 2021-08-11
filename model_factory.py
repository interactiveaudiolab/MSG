from models.MelGAN import GeneratorMel, GeneratorMelMix, DiscriminatorMel, SpecDiscriminator
from models.Demucs import Demucs

class ModelFactory():
    def __init__(self, model, use_mix, multi_disc):
        self.model = model
        self.use_mix = use_mix
        self.multi_disc = multi_disc
    def generator(self, *args, **kwargs):
        if self.model == "melgan":
            if self.use_mix:
                return GeneratorMelMix(args,kwargs)
            else:
                return GeneratorMel(args,kwargs)
        if self.model == "demucs":
            return Demucs(args,kwargs)
        else:
            raise ValueError('Invalid Model')
    def discriminator(self, in_channels, *args, **kwargs):
        if self.multi_disc:
            return DiscriminatorMel(args,kwargs), SpecDiscriminator(in_channels)
        else:
            return DiscriminatorMel(args,kwargs)
