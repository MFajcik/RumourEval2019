from task_A.models.baseline import Baseline
from task_A.models.baseline_lstm import Baseline_LSTM
from task_A.frameworks.base_framework import Base_Framework
from task_A.models.sel_att_and_baseline import SelfAttandBsline
from task_A.models.self_attention_text_only import SelAttTextOnly
from task_A.frameworks.text_features_framework import Text_Feature_Framework
from task_A.frameworks.text_framework import Text_Framework

__author__ = "Martin Fajčík"


class SolutionA:
    def __init__(self, config):
        self.config = config

    def create_model(self):
        """
        Create and validate model
        """
        modelf = None
        fworkf = Base_Framework
        if self.config["active_model"] == "baseline_LSTM":
            # 204 804 params
            modelf = Baseline_LSTM
        elif self.config["active_model"] == "baseline":
            modelf = Baseline
        elif self.config["active_model"] == "selfatt_textonly":
            modelf = SelAttTextOnly
            fworkf = Text_Framework
        elif self.config["active_model"] == "selfatt_text_and_baseline":
            modelf = SelfAttandBsline
            fworkf = Text_Feature_Framework

        modelframework = fworkf(self.config["models"][self.config["active_model"]])
        modelframework.train(modelf)

    def submit_model(self, model):
        """
        Load model and run submission
        """
        pass
