#!/bin/bash

echo "Running Student Performance Analysis..."
cd student_performance
python3 student_performance_analysis.py
cd ..

echo "\nRunning Adults Income Analysis..."
cd adults_income
python3 adults_income_analysis.py
cd ..

echo "\nRunning Synthetic Data Analysis..."
cd synthetic_data
python3 synthetic_data.py
cd ..

echo "\nRunning GANITE Counterfactual Analyses..."
echo "Running Student Performance GANITE..."
cd student_performance/counterfactuals
python3 main_ganite.py
cd ../..

echo "Running Adults Income GANITE..."
cd adults_income/counterfactuals
python3 main_ganite.py
cd ../..

echo "Running Synthetic Data GANITE..."
cd synthetic_data/counterfactuals
python3 main_ganite.py
cd ../..

echo "\nAnalysis complete. Check results in respective folders and counterfactuals/ subdirectories."