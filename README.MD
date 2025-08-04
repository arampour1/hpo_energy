# ⚡ Energy-aware Hyperparameter Optimization for DistilBERT

This project performs **multi-objective hyperparameter optimization** for fine-tuning DistilBERT on the SST-2 dataset (GLUE benchmark), targeting both:
- 🔹 High **macro-F1** score  
- 🔸 Low **energy consumption (kWh)**

Supported optimization algorithms:
- ✅ Genetic Algorithm (GA)
- ✅ Particle Swarm Optimization (PSO)
- ✅ Bayesian Optimization (BO)

Results are saved in the `./results/` folder as CSV and visualized in PNG plots.

---

## 📦 Requirements

Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# OR
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

If you plan to run on GPU, ensure you use the **CUDA-enabled PyTorch build**.  
See notes inside `requirements.txt`.

---

## 🚀 Running the Project

### Basic usage:

```bash
python main.py --method ga --pop 5 --ngen 4 --gpu
python main.py --method pso --method pso --pop 5 --ngen 4 --alpha 0.02 --gpu
python main.py --method bo --method bo --trials 20 --gpu
```

### Optional arguments:

| Argument      | Description                                  | Default |
|---------------|----------------------------------------------|---------|
| `--method`    | Optimization method: `ga`, `pso`, or `bo`    | `ga`    |
| `--gpu`       | Use GPU if available                         | `False` |
| `--seed`      | Random seed                                  | `42`    |
| `--pop`       | Population size for GA/PSO                   | `8`     |
| `--ngen`      | Generations for GA/PSO                       | `5`     |
| `--trials`    | Number of trials for BO                      | `15`    |
| `--alpha`     | PSO energy penalty factor                    | `0.02`  |

---

## 📂 Output Files

All results are saved in `./results/`, including:
- CSV of final configurations (`*_results.csv`)
- Scatter plots:
  - Energy vs F1
  - F1 vs Learning Rate
  - F1 vs Batch Size
  - F1 vs Epochs

---

## 🔧 Notes

- Training uses a small subset of SST-2 for fast prototyping.
- Energy tracking is handled via the [CodeCarbon](https://mlco2.github.io/codecarbon/) library.
- This is research code for comparing optimization strategies, not production-ready training.

---

## 📜 License

Advance Optimization
Summer 2025

Group 1
Amirhossein Rampour, Saroosh Ashraf, Hammad Cheema

