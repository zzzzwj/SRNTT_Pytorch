from utils.config import get_config
from solver.refsrsolver import RefSRSolver

if __name__ == '__main__':
    cfg = get_config('config/srntt_vgg19_div2k.yml')
    solver = RefSRSolver(cfg)

    solver.run()
