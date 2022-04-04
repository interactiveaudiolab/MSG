from models.MelGAN import GeneratorMel, GeneratorMelMix, DiscriminatorMel, SpecDiscriminator
import numpy as np
from models.Demucs import Demucs
import torch
from models.HiFiGAN import Discriminator


class ModelFactory():
    def __init__(self, config, optim=None):
        self.config = config
        self.optim = optim
    def generator(self):
        if self.config.model == "melgan":
            if self.config.use_mix:
                return GeneratorMelMix(self.config.n_mel_channels, self.config.ngf, self.config.n_residual_layers,self.config.skip_cxn)
            else:
                return GeneratorMel(self.config.n_mel_channels, self.config.ngf, self.config.n_residual_layers,self.config.skip_cxn)
        if self.config.model == "demucs":
            return Demucs([self.config.source],audio_channels=self.config.audio_channels, samplerate=self.config.sample_rate ,segment_length=int(self.config.segment_duration * self.config.sample_rate), skip_cxn = self.config.skip_cxn)
        else:
            raise ValueError('Invalid Model')
    def discriminator(self):
        if self.config.multi_disc:
            return DiscriminatorMel(self.config.num_D, self.config.ndf, self.config.n_layers_D, self.config.downsamp_factor), SpecDiscriminator(self.config.n_mel_channels)
        else:
            return Discriminator()


class MultiSpecDiscriminator():
    def __init__(self, in_channels, splits, optimizer, config):
        super().__init__()
        self.in_channels= in_channels
        self.splits = splits
        self.optimizer = optimizer
        self.model_dict = {}
        self.optimizer_dict = {}
        self.config = config
        self.create_splits()

    def create_splits(self):
        for i in self.splits:
            curr_models = [SpecDiscriminator(np.ceil(self.in_channels/i).astype(int)) for k in range(i)]
            curr_optimizers = [self.optimizer(curr_models[k].parameters(),lr=self.config.lr, betas=(self.config.b1,self.config.b2)) for k in range(i)]
            self.model_dict[i] = curr_models
            self.optimizer_dict[i] = curr_optimizers

    def spectrogram(self, x):
        window_length = (self.in_channels - 1) * 2
        hop_length = window_length // 4
        spec = torch.stft(
            x,
            n_fft=window_length,
            hop_length=hop_length,
            win_length=window_length,
            center=True,
        )

        real_part, imag_part = spec.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        magnitude += 1e-5

        return torch.log10(magnitude)

    def forward(self,x):
        # TODO make sure spectrogram is taken over right dimension
        s = self.spectrogram(x)
        outputs = []
        for key, disc in self.model_dict.items():
            chunk_size = np.ceil(s.shape[1]/key).astype(int)
            disc_outputs = []
            for i in range(key):
                curr_spec = s[:,i*chunk_size:(i+1)*chunk_size,:]
                if i == key-1 and s.shape[1]%key !=0:
                    pad = torch.nn.ZeroPad2D((0,s.shape[1]%key,0,0))
                    curr_spec =  pad(curr_spec)
                disc_outputs.append(disc[i](curr_spec))
            outputs.append(disc_outputs)
        return outputs

    def optimizer_step(self):
        for value in self.optimizer_dict.values():
            for opt in value:
                opt.step()

    def to(self, device):
        for value in self.model_dict.values():
            for model in value:
                model.to(device)

    def train(self):
        for value in self.model_dict.values():
            for model in value:
                model.train()

    def zero_grad(self):
        for value in self.model_dict.values():
            for model in value:
                model.train()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _get(self):
        return self.model_dict, self.optimizer_dict

