import argparse
import train
import Create_Summary
def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_config', '-t', type=str, help='Experiment yaml file',required=True)
        parser.add_argument('--eval_config', '-e', type=str, help='Experiment yaml file',required=True)
        exp, exp_args = parser.parse_known_args()
        print(exp_args)
        run_url, best_g  = train.train(exp.train_config)
        Create_Summary.create_summary(run_url,best_g)

if __name__ == '__main__':
    main()
