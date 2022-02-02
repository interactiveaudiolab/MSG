import argparse
import train
import Create_Summary
def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', '-e', type=str, help='Experiment yaml file',required=True)
        exp, exp_args = parser.parse_known_args()
        run_url, best_g  = train.train(exp.config)
        Create_Summary.create_summary(run_url,best_g,config=exp.config)

if __name__ == '__main__':
    main()
