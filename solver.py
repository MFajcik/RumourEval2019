import json
import logging
import os
import sys

from solutionsA import SolutionA
from utils import setup_logging

__author__ = "Martin Fajčík"


class TaskSolver():
    def __init__(self, config: dict):
        self.config = config

    def solvetask(self, taskname: str):
        if taskname == "A":
            solution = SolutionA(self.config)
            solution.create_model()


if __name__ == "__main__":
    Experiment_Name = "Bert_nofeats"
    with open("configurations/config.json") as conffile:
        config = json.load(conffile)
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  extra_name=Experiment_Name,
                  logpath="logs/",
                  config_path="configurations/logging.yml")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.debug("Configuration:\n" + json.dumps(config, indent=4))
    solver = TaskSolver(config)
    solver.solvetask("A")
