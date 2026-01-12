class Metric:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
    def avg(self):
        return self.sum / self.count if self.count != 0 else 0.0