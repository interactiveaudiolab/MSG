"""
Evaluate the run and write to a csv file with the following format:
Experiment name, SD-SDR, SI-SDR, SNR, SIR, SAR, Developer Sentiment
"""
import argparse
import utils.RunEvaluation as RE
def create_summary(wandb_url,best_g, name=None, config=None, sentiment='', notes=''):
    # run Evaluate
    # request user feedback
    if config:
        demucs_medians, msg_medians = RE.Evaluate(exp.config,best_g)
        msg_results = [name, msg_medians[0], msg_medians[1], msg_medians[2],
                sentiment, notes, wandb_url]
    else:   
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', '-n', type=str, help='Experiment yaml file',
                            required=True)
        parser.add_argument("--sentiment", "-s", type=str,
                            help='Developer sentiment for the model\'s performance '
                                'ranked on a scale of 0 - 10', required=True)
        parser.add_argument("--notes", type=str,
                            help='Any additional notes the developer may have',
                            required=False)
        parser.add_argument("--config", "-c", type=str,
                            help="Parameter management file", required=True)
        exp, exp_args = parser.parse_known_args()
        

        demucs_medians, msg_medians = RE.Evaluate(exp.config)
        msg_results = [exp.name, msg_medians[0], msg_medians[1], msg_medians[2],
                exp.sentiment, exp.notes, wandb_url]

    titles = ['Experiment', 'BSSEval SDRv4', 'SAR', 'SIR',
            'Developer Sentiment 0 (worst) - 10 (best)', 'notes', 'Run URL']

    demucs_results = ["Demucs Baseline", demucs_medians[0], demucs_medians[1],
                    demucs_medians[2], "5",
                    "demucs is the baseline, use its sentiment score as a reference value", '']

    # write outputs to CSV
    with open('summary.csv', 'w') as f:
        for item in titles:
            f.write(item+', ') if item != titles[-1] else f.write(item)
        f.write('\n')
        for result in demucs_results:
            result = str(result).replace(',', ';')
            f.write(result+', ') if result != demucs_results[-1] else f.write(result)
        f.write('\n')
        for result in msg_results:
            result = str(result).replace(',', ';')
            f.write(result+', ') if result != msg_results[-1] else f.write(result)
        f.write('\n')
if __name__ == '__main__':
    create_summary('', '')
