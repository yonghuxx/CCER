# hy python学习
# 编程时间:2024-12-31 22:17
from metrics import MetricScore, args


class Experiments:
    def __init__(self):
        self.metrics = MetricScore()

    def get_score(self):
        return self.metrics.get_score()


def score():
    experiments = Experiments()
    return experiments.get_score()