from easydict import EasyDict as edict

def get_config():
    config = edict()
    config.url = "http://127.0.0.1:8000"
    config.data_folder = "C:/data/samsung_sem/open/test/SEM"
    return config