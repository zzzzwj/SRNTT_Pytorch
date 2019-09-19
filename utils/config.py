import re
import yaml


def get_config(cfg_path):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=loader)
    return cfg


def save_config(info, cfg_path):
    with open(cfg_path, 'w') as f:
        yaml.dump(info, f, Dumper=yaml.SafeDumper)


if __name__ == '__main__':
    config = get_config('config/srntt_vgg19_div2k.yml')
    print(config)
