from task_A.frameworks.base_framework import Base_Framework
from task_A.frameworks.base_framework_seq import Base_Framework_SEQ
from task_A.frameworks.bert_framework import BERT_Framework
from task_A.frameworks.bert_framework_with_f import BERT_Framework_with_f
from task_A.frameworks.text_features_framework import Text_Feature_Framework
from task_A.frameworks.text_framework_branch import Text_Framework
from task_A.frameworks.text_framework_seq import Text_Framework_Seq
from task_A.models.baseline import Baseline
from task_A.models.baseline_lstm import Baseline_LSTM
from task_A.models.bert_with_features import BertModelForStanceClassificationWFeatures
from task_A.models.sel_att_and_baseline import SelfAttandBsline
from task_A.models.self_attention_text_only import SelAttTextOnly, SelAttTextOnlyWithoutPrepInput
from task_A.models.text_BERT import BertModelForStanceClassification

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
        elif self.config["active_model"] == "selfatt_textonly_seq":
            modelf = SelAttTextOnlyWithoutPrepInput
            fworkf = Text_Framework_Seq
        elif self.config["active_model"] == "selfatt_text_and_baseline":
            modelf = SelfAttandBsline
            fworkf = Text_Feature_Framework
        elif self.config["active_model"] == "BERT_textonly":
            modelf = BertModelForStanceClassification
            fworkf = BERT_Framework
        elif self.config["active_model"] == "features_seq":
            modelf = Baseline
            fworkf = Base_Framework_SEQ
        elif self.config["active_model"] == "BERT_withf":
            modelf = BertModelForStanceClassificationWFeatures
            fworkf = BERT_Framework_with_f

        modelframework = fworkf(self.config["models"][self.config["active_model"]])
        modelframework.train(modelf)

    def submit_model(self, model):
        """
        Load model and run submission
        """
        pass
