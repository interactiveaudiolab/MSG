import numpy as np
import wandb
import sys
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import imageio
import torch

def save_model(save_path, netG, netD, optG, optD ,epoch, spec, netD_spec, optD_spec, config):
    '''
    Input: save_path (string), netG (state_dict), netD (state_dict),
    optG (torch.optim), optD(torch.optim)
    '''
    print("saving models")
    torch.save(netG, save_path +  str(epoch) + "netG.pt")
    torch.save(netD, save_path  +  str(epoch) + "netD.pt")
    torch.save(optG, save_path   +  str(epoch) + "optG.pt")
    torch.save(optD, save_path   +  str(epoch) + "optD.pt")
    if spec:
        torch.save(netD_spec, save_path  +  str(epoch) + "netD_spec.pt")
        torch.save(optD_spec, save_path   +  str(epoch) + "optD_spec.pt")


def log_writer(writer, costs, steps):
    writer.add_scalar("loss/discriminator", costs[-1][0], steps)
    writer.add_scalar("loss/generator", costs[-1][1], steps)
    writer.add_scalar("loss/feature_matching", costs[-1][2], steps)
    writer.add_scalar("loss/mel_reconstruction", costs[-1][3], steps)
    writer.add_scalar("loss/sdr", costs[-1][4], steps)
    return None


def stdout_writer(epoch, iterno, costs):
    sys.stdout.write(f'\r[Epoch {epoch}, Batch {iterno}]:\
                        [Generator Loss: {costs[-1][1]:.4f}]\
                        [Discriminator Loss: {costs[-1][0]:.4f}]\
                        [Feature Loss: {costs[-1][2]:.4f}]\
                        [Reconstruction Loss: {costs[-1][3]:.4f}]\
                        [SDR: {costs[-1][4]:.4f}]')

def validation_writer(epoch, steps):
    sys.stdout.write(f'\r Validtation Step: Epoch {epoch}, Step {steps}')

def wandb_writer(epoch, costs):
    wandb.log({
        'Generator Loss': costs[-1][1],
        'Wav Discriminator Loss': costs[-1][0],
        'Spec Discriminator Loss': costs[-1][5],
        'Wav Feature Loss': costs[-1][2],
        'Spec Feature Loss': costs[-1][6],
        'Reconstruction Loss': costs[-1][3],
        'SDR': costs[-1][4],
        'epoch': epoch
    })




def basic_logs(costs, writer, steps, epoch, iterno):
    log_writer(writer, costs, steps)
    stdout_writer(epoch, iterno, costs)
    wandb_writer(epoch, costs)


def iteration_logs(netD, netG, optG, optD, netD_spec, optD_spec,
                   steps, epoch, config, best_SDR, best_reconstruct, aud, costs):
    ######################
    # Update tensorboard #
    ######################


    if epoch == 0:
        sf.write('validation_original.wav', np.transpose(aud[1]),
                 config.sample_rate)
        sf.write('validation_demucs.wav', np.transpose(aud[0]),
                 config.sample_rate)
        wandb.log({"Validation Audio":
            [wandb.Audio(
                'validation_original.wav',
                caption="Original Sample",
                sample_rate=config.sample_rate
            ),
                wandb.Audio(
                    'validation_demucs.wav',
                    caption="Demucs Sample",
                    sample_rate=config.sample_rate
                )]})
    if costs[-1][4] > best_SDR and not config.disable_save:
        save_model(config.model_save_dir, netG.state_dict(),
                   netD.state_dict(), optG, optD, epoch, spec=True,
                   netD_spec=netD_spec.state_dict(), optD_spec=optD_spec,
                   config=config)
        best_SDR = costs[-1][4]
    if costs[-1][3] < best_reconstruct and not config.disable_save:
        save_model(config.model_save_dir, netG.state_dict(),
                   netD.state_dict(), optG, optD, epoch, spec=True,
                   netD_spec=netD_spec.state_dict(), optD_spec=optD_spec,
                   config=config)
        best_reconstruct = costs[-1][3]
    wandb.log({
        'Valid Generator Loss': costs[-1][1],
        'Valid Discriminator Loss': costs[-1][0],
        'Valid Feature Loss': costs[-1][2],
        'Valid Reconstruction Loss': costs[-1][3],
        'Valid SDR': costs[-1][4]
    })
    sf.write(f'generated_{steps}.wav', np.transpose(aud[2]),
             config.sample_rate)
    wandb.log({f'{steps} Steps':
        [wandb.Audio(
            f'generated_{steps}.wav',
            caption=f'Generated Audio, {steps} Steps',
            sample_rate=44100
        )]
    })

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7))

    titles = ['Demucs', 'Original', 'MSG']
    axes = [ax1, ax2, ax3]

    for i in range(3):
        if not config.mono:
            aud_mono = (aud[i][0, :] + aud[i][1, :])
            aud_mono /= np.max(np.abs(aud_mono))
            spec_data = librosa.feature.melspectrogram(y=aud_mono,
                                                       sr=44100,
                                                       n_mels=128,
                                                       fmax=8000)
        else:
            spec_data = librosa.feature.melspectrogram(y=aud[i], sr=44100,
                                                       n_mels=128,
                                                       fmax=8000)
        S_dB = librosa.power_to_db(spec_data, ref=np.max)
        _fig_ax = librosa.display.specshow(S_dB, x_axis='time',
                                           y_axis='mel',
                                           sr=config.sample_rate,
                                           fmax=8000, ax=axes[i])
        axes[i].set(title=titles[i])
    plt.savefig(f'spectrogram_{steps}.png')
    wandb.log({f'Spectrograms, {steps} Steps':
                   [wandb.Image(
                       imageio.imread(f'spectrogram_{steps}.png'))]})