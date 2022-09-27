from easydict import EasyDict as edict

def get_config():
    config = edict()
    config.args = "trained/22_09_13_18_20_20/argument.json"
    config.weight = "weight.pt"
    config.device = "cpu"
    return config