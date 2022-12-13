"""
Evaluate the run and write to a csv file with the following format:
Experiment name, SD-SDR, SI-SDR, SNR, SIR, SAR, Developer Sentiment
"""
import argparse
import utils.RunObjectiveEval as RE
import os
import json
def create_summary():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str,
                            help="Parameter management file", required=True)
    exp, exp_args = parser.parse_known_args()
    RE.Evaluate(exp.config, '')

if __name__ == '__main__':
    create_summary()
