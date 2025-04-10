from easydict import EasyDict 
import yaml

def load_config(cfg_filename):
    with open(cfg_filename, 'r') as f :
        config = yaml.safe_load(f)
    return EasyDict(config)


if __name__ == '__main__':
    cfg = load_config('config/model_cfg.yaml')