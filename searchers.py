"""
Search strategies: GA, PSO, BO
"""

import random
from config import HYPER_SPACE
from data_model import evaluate_config

# ----- Genetic Algorithm (GA) -----
def ga_search(args, device):
    from deap import base, creator, tools, algorithms
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    tb = base.Toolbox()
    tb.register("lr_gene", random.uniform, *HYPER_SPACE["lr"])
    tb.register("bs_gene", lambda: random.choice(HYPER_SPACE["bs"]))
    tb.register("ep_gene", lambda: random.randint(*HYPER_SPACE["epoch"]))
    tb.register("individual", tools.initCycle, creator.Individual,
                (tb.lr_gene, tb.bs_gene, tb.ep_gene), 1)
    tb.register("population", tools.initRepeat, list, tb.individual)

    def deap_eval(ind):
        bs = min(HYPER_SPACE["bs"], key=lambda x: abs(x - int(round(ind[1]))))
        f1, e = evaluate_config(ind[0], bs, int(round(ind[2])), seed=args.seed, device=device)
        return f1, e
    
    def safe_mutate(ind):
        tools.mutGaussian(ind, mu=0, sigma=0.5e-5, indpb=0.2)
        ind[1] = min(HYPER_SPACE["bs"], key=lambda x: abs(x - int(round(ind[1]))))
        ind[2] = max(HYPER_SPACE["epoch"][0], min(HYPER_SPACE["epoch"][1], int(round(ind[2]))))
        return ind,

    tb.register("evaluate", deap_eval)
    tb.register("mate", tools.cxBlend, alpha=0.5)
    tb.register("mutate", safe_mutate)
    tb.register("select", tools.selNSGA2)

    pop = tb.population(n=args.pop)
    hof = tools.HallOfFame(20)
    algorithms.eaMuPlusLambda(pop, tb, mu=args.pop, lambda_=args.pop, cxpb=0.5, mutpb=0.3,
                              ngen=args.ngen, halloffame=hof, verbose=True)

    return [{"method": "GA", "lr": i[0], "bs": i[1], "ep": i[2],
             "f1": i.fitness.values[0], "energy": i.fitness.values[1]} for i in hof]

# ----- Particle Swarm Optimization (PSO) -----
def pso_search(args, device):
    import pyswarms as ps
    import numpy as np
    res = []

    def pso_obj(x):
        α = args.alpha
        out = []
        for lr, bs_idx, ep in x:
            lr = float(lr)
            bs_idx = int(np.clip(round(bs_idx), 0, len(HYPER_SPACE["bs"]) - 1))
            bs = HYPER_SPACE["bs"][bs_idx]
            ep = int(np.clip(round(ep), *HYPER_SPACE["epoch"]))
            f1, e = evaluate_config(lr, bs, ep, seed=args.seed, device=device)
            out.append(-(f1 - α * e))
        return np.array(out)

    bounds = (
        np.array([HYPER_SPACE["lr"][0], 0, HYPER_SPACE["epoch"][0]]),
        np.array([HYPER_SPACE["lr"][1], len(HYPER_SPACE["bs"]) - 1, HYPER_SPACE["epoch"][1]])
    )
    pso = ps.single.GlobalBestPSO(n_particles=args.pop, dimensions=3, bounds=bounds,
                                  options={'c1': 1.5, 'c2': 1.5, 'w': 0.7})
    pso.optimize(pso_obj, iters=args.ngen)

    for p in pso.swarm.position:
        lr, bs_idx, ep = p
        bs_idx = int(np.clip(round(bs_idx), 0, len(HYPER_SPACE["bs"]) - 1))
        bs = HYPER_SPACE["bs"][bs_idx]
        ep = int(np.clip(round(ep), *HYPER_SPACE["epoch"]))
        f1, e = evaluate_config(lr, bs, ep, seed=args.seed, device=device)
        res.append({"method": "PSO", "lr": lr, "bs": bs, "ep": ep, "f1": f1, "energy": e})
    return res

# ----- Bayesian Optimization (BO) -----
def bo_search(args, device):
    import optuna
    def objective(trial):
        lr = trial.suggest_float("lr", *HYPER_SPACE["lr"], log=True)
        bs = trial.suggest_categorical("bs", HYPER_SPACE["bs"])
        ep = trial.suggest_int("ep", *HYPER_SPACE["epoch"])
        return evaluate_config(lr, bs, ep, seed=args.seed, device=device)

    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=args.trials)
    return [{"method": "BO", "lr": t.params["lr"], "bs": t.params["bs"], "ep": t.params["ep"],
             "f1": t.values[0], "energy": t.values[1]} for t in study.best_trials]
