"""
Evaluate the run and write to a csv file with the following format:
Experiment name, SD-SDR, SI-SDR, SNR, SIR, SAR, Developer Sentiment
"""
import argparse

# run Evaluate
SDR, SIR, SAR = 0, 0, 0
# request user feedback
parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', type=str, help='Experiment yaml file',
                    required=True)
parser.add_argument("--sentiment", "-s", type=str,
                    help='Developer sentiment for the model\'s performance '
                         'ranked on a scale of 0 - 10', required=True)
parser.add_argument("--notes", type=str,
                    help='Any additional notes the developer may have',
                    required=False)
exp, exp_args = parser.parse_known_args()

titles = ['Experiment', 'BSSEval SDRv4', 'SIR', 'SAR', 'Developer Sentiment 0-10', 'notes']
results = [exp.name, SDR, SIR, SAR, exp.sentiment, exp.notes]
# write outputs to CSV
with open('summary.csv', 'w') as f:
    for item in titles:
        f.write(item+', ') if item != titles[-1] else f.write(item)
    f.write('\n')
    for result in results:
        result = str(result).replace(',', ';')
        f.write(result+', ') if result != results[-1] else f.write(result)
    f.write('\n')
