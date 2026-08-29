import os
import sys

def run_all():
    print("==================================================")
    print("ANTICIPATORY ROUTING: FULL REPRODUCIBILITY SCRIPT")
    print("==================================================")
    
    print("\n[1/4] Validating Grid Scale and Anticipation Window...")
    os.system(f"{sys.executable} experiments/validate_scale.py")
    
    print("\n[2/4] Validating Kinematic Wave Dynamics & Generating Plots...")
    os.system(f"{sys.executable} experiments/validate_kinematic_wave.py")
    
    print("\n[3/4] Running Inference Latency Benchmarks...")
    os.system(f"{sys.executable} experiments/benchmark_inference.py")
    
    print("\n[4/4] Executing Final Correlation Sweep (Training & Evaluation)...")
    print("      Note: This may take ~10-20 minutes depending on hardware.")
    os.system(f"{sys.executable} experiments/run_sweep.py")
    
    print("\n==================================================")
    print("ALL EXPERIMENTS COMPLETE.")
    print("Results saved to: results/")
    print("Figures saved to: figures/")
    print("==================================================")

if __name__ == "__main__":
    run_all()
