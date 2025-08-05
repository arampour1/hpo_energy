"""
Main entry for Energy-aware multi-objective HPO.
Usage:
    python main.py --method ga --gpu
    python main.py --method pso --seed 123
    python main.py --method bo --trials 30
"""

import argparse
import torch
import warnings
from config import set_seed
from searchers import ga_search, pso_search, bo_search
from visualize import save_all_outputs

def main():
    parser = argparse.ArgumentParser(description="Energy-aware HPO")
    parser.add_argument("--method", choices=["ga","pso","bo"], default="ga")
    parser.add_argument("--pop",    type=int, default=8,   help="population / swarm size")
    parser.add_argument("--ngen",   type=int, default=5,   help="GA/PSO generations / iterations")
    parser.add_argument("--trials", type=int, default=15,  help="BO trial count")
    parser.add_argument("--alpha",  type=float, default=0.02, help="energy penalty for PSO")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--gpu",    action="store_true", help="use CUDA if available")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    if args.method == "ga":
        results = ga_search(args, device)
    elif args.method == "pso":
        results = pso_search(args, device)
    else:
        results = bo_search(args, device)

    save_all_outputs(results, args.method.upper())

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
