# Machine Learning Analysis Pipeline

## How to Run

Execute the analysis pipeline:

```bash
bash run_analysis.sh
```

This will:
1. Run Random Forest regression on student performance data (G3 prediction)
2. Generate MSE metrics and visualizations
3. Save results as `student_performance_results.png` in the `student_performance/` folder
4. Run Random Forest classification on adults income data
5. Generate accuracy metrics and visualizations
6. Save results as `adults_income_results.png` in the `adults_income/` folder
7. Run Random Forest regression on synthetic causal data
8. Generate MSE metrics and comprehensive visualizations
9. Save results as `synthetic_data_results.png` in the `synthetic_data/` folder
10. Run GANITE counterfactual analysis on student performance data
11. Generate counterfactual outcomes and correlation plots in `student_performance/counterfactuals/alpha_*/`
12. Run GANITE counterfactual analysis on adults income data
13. Generate counterfactual outcomes and correlation plots in `adults_income/counterfactuals/alpha_*/`
14. Run GANITE counterfactual analysis on synthetic data
15. Generate counterfactual outcomes and correlation plots in `synthetic_data/counterfactuals/alpha_*/`

## GANITE Counterfactual Analysis

Each dataset includes GANITE (Generative Adversarial Nets for Individualized Treatment Effects) analysis:

- **Training**: 10,000 iterations across 6 alpha values (0-5)
- **Output**: Organized in `alpha_*` folders with CSV data and correlation plots
- **Evaluation**: `optimal_alpha_ganite.txt` lists configurations with correlation > 0.7
- **Treatment Variables**: 
  - Student Performance: Gender (sex)
  - Adults Income: Gender (SEX)
  - Synthetic Data: Simulated binary treatment