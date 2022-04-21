import os

cwd = os.getcwd()

os.system("python aquire_best_spectrum.py --meta fc --sample H2OCu")
os.system("python DRE_experiments.py --meta fc --sample H2OCu")
os.system("python DRE_experiments.py --meta none --sample H2OCu")
os.system("python comparison_criterion.py --meta fc --sample H2OCu")
os.system("python comparison_iterations.py --meta fc --sample H2OCu")
os.system("python DRE_experiments.py --meta linear --sample H2OCu")
os.system("python DRE_experiments.py --meta average --sample H2OCu")
os.system("python DRE_experiments.py --meta none_tuned --sample H2OCu")