import torch

from model.predict import PredictModel


config = {
    "max_features": 5,
    "embedding_size": 3,
    "prediction_hidden_width": [3, 2],
    "keep_prob": 0.8,
    "lr": 1e-5,
    "decay_step": 1e5,
    "l2_reg": 3,
    "model_name": "PNN",
}
pm = PredictModel(config)
print(pm.model)

input = torch.rand(10, 100) * 5
input = input.int()
print(input)
output = pm(input)
print(output)

config2 = {
    "max_features": 5,
    "embedding_size": 3,
    "lr": 1e-5,
    "decay_step": 1e5,
    "l2_reg": 3,
    "model_name": "FM",
}
pm2 = PredictModel(config2)
# output2 = pm2(input)
# print(output2)

y = torch.randint_like(output, 0, 2)
