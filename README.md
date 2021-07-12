## Last-Mile Imputation with MelGAN

This model uses MelGAN (Kumar et. al. 2018) to denoise and reconstruct the output of audio source separators. Our current model works on the output of Demucs, trained on each source individually. Training code can be found in the experiments directory and our implementation of MelGAN can be found in the models directory.