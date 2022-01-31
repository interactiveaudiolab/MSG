import argparse
import train
import Create_Summary
def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_config', '-t', type=str, help='Experiment yaml file',required=True)
        parser.add_argument('--eval_config', '-e', type=str, help='Experiment yaml file',required=True)
        exp, exp_args = parser.parse_known_args()
        run_url = train.train(exp.train_config)
        Create_Summary.create_summary(run_url)
