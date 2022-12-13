"""
Evaluate the run and write to a csv file with the following format:
Experiment name, SD-SDR, SI-SDR, SNR, SIR, SAR, Developer Sentiment
"""
import argparse
import utils.RunEvaluation as RE
import os
def create_summary(wandb_url,best_g, names=None, config=None, sentiment='', notes=''):
    # run Evaluate
    # request user feedback
    if config:
        medians = RE.Evaluate(config,best_g,names)
        results = []
        for i in range(len(medians)):
            results.append([names[i], medians[i][0], medians[i][1], medians[i][2],
                sentiment, notes, wandb_url])
    else:   
        parser = argparse.ArgumentParser()
        parser.add_argument("--notes", type=str,
                            help='Any additional notes the developer may have',
                            required=False)
        parser.add_argument("--config", "-c", type=str,
                            help="Parameter management file", required=True)
        exp, exp_args = parser.parse_known_args()
        
        RE.Evaluate(exp.config, ['msg_bass'])

if __name__ == '__main__':
    create_summary('', '')
