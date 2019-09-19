from data.div2k import DIV2K
from data.benchmark import Benchmark


def getDataSet(cfg, split, scale):
    if cfg['name'].lower() == 'div2k':
        return DIV2K(cfg, split, scale)
    elif 'set' in cfg['name'].lower():
        return Benchmark(cfg, split, scale)
    else:
        raise ValueError('Dataset {} is not supported!'.format(cfg['name']))
