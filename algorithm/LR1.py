# LR1 - Linear Regression Model 1

class LinearRegressionModel:
    def __init__(self, m ,b):
        self.m = m
        self.b = b

    def predict(self,study_time):
        return self.m * study_time + self.b # LR function (y = mx+b)


m = 4.69
b = 41.77

model = LinearRegressionModel(m,b)