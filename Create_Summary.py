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
        parser.add_argument("--best_checkpoints", "-b", type=str, 
                            help="absolute path to the best checkpoints", required=True)
        parser.add_argument("--evaluation_model", "-m", type=str, 
                            help="What model we are evaluating", required=True)
        exp, exp_args = parser.parse_known_args()
        
        best_checkpoints = [exp.best_checkpoints+ os.sep + elem for elem in os.listdir(exp.best_checkpoints)]
        best_names = [elem.replace('netG.pt','') for elem in os.listdir(exp.best_checkpoints)]
        best_names.insert(0,exp.evaluation_model)
        medians = RE.Evaluate(exp.config, best_checkpoints,best_names)
        results = []
        for i in range(len(medians)):
            results.append([best_names[i], medians[i][0], medians[i][1], medians[i][2],
                sentiment, notes, wandb_url])

    titles = ['Experiment', 'BSSEval SDRv4', 'SAR', 'SIR',
            'Developer Sentiment 0 (worst) - 10 (best)', 'notes', 'Run URL']



    # write outputs to CSV
    with open('summary.csv', 'w') as f:
        for item in titles:
            f.write(item+', ') if item != titles[-1] else f.write(item)
        f.write('\n')
        for result in results:
            for elem in result:
                r = str(elem).replace(',', ';')
                f.write(r +', ') if elem != result[-1] else f.write(r)
            f.write('\n')
if __name__ == '__main__':
    create_summary('', '')
