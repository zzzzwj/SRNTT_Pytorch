import yaml


def get_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg


if __name__ == '__main__':
    config = get_config('config/srntt_vgg19_div2k.yml')
    print(config)
