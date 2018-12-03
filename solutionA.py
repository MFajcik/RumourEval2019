from task_A_models.baseline import Baseline
from task_A_models.baseline_lstm import Baseline_LSTM
from task_A_models.modelframework import BaseFramework
from task_A_models.sel_att_and_baseline import SelfAttandBsline
from task_A_models.self_attention_text_only import SelAttTextOnly

__author__ = "Martin Fajčík"


class SolutionA:
    def __init__(self, config):
        self.config = config

    def create_model(self):
        """
        Create and validate model
        """
        modelf = None
        fworkf = BaseFramework
        if self.config["active_model"] == "baseline_LSTM":
            # 204 804 params
            modelf = Baseline_LSTM
        elif self.config["active_model"] == "baseline":
            modelf = Baseline
        elif self.config["active_model"] == "selfatt_textonly":
            # 1 653 004 params
            modelf = SelAttTextOnly
        elif self.config["active_model"] == "selfatt_text_and_baseline":
            modelf = SelfAttandBsline

        modelframework = fworkf(self.config["models"][self.config["active_model"]])
        modelframework.train(modelf)

    def submit_model(self, model):
        """
        Load model and run submission
        """
        pass
