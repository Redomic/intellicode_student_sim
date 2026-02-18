"""
IntelliT Full Evaluation Runner

This script runs the complete evaluation pipeline:
1. Runs the standard ADAPTIVE simulation (IntelliCode)
2. Runs the STATELESS baseline simulation
3. Generates separate metrics and trajectory files for each
4. Prints a comparison summary

Usage:
    python run_full_eval.py [--personas 10] [--concurrent 2]
"""

import sys
import os
import asyncio
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '../backend'))

from src.intellit_adapter import load_strivers_a2z_questions
from src.persona_generator import generate_diverse_personas
from src.db_setup import create_synthetic_users
from src.simulation_engine import run_parallel_simulations
from src.metrics_tracker import MetricsTracker
from src.simulation_logger import init_logger, finalize_logger

# Configuration
CONFIG = {
    "n_personas": 10,  # Default, can be overridden by args
    "n_questions": 200,
    "max_concurrent": 2,
    "seed": 42,
    "base_results_dir": "results_full_eval"
}

def print_banner(text):
    print(f"\n{'='*80}")
    print(f"{text:^80}")
    print(f"{'='*80}\n")

async def run_simulation_phase(mode, personas, questions, output_dir):
    """Run a single simulation phase (adaptive or baseline)."""
    stateless = (mode == "baseline")
    print_banner(f"PHASE: {mode.upper()} SIMULATION")
    
    trajectories_dir = os.path.join(output_dir, "trajectories")
    metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
    os.makedirs(trajectories_dir, exist_ok=True)
    
    print(f"Mode: {'Stateless Baseline' if stateless else 'Adaptive IntelliCode'}")
    print(f"Output: {output_dir}")
    
    # Run simulations
    results = await run_parallel_simulations(
        personas=personas,
        questions=questions,
        max_concurrent=CONFIG['max_concurrent'],
        stateless=stateless,
        output_dir=trajectories_dir
    )
    
    # Calculate metrics
    print(f"\nCalculating metrics for {mode}...")
    tracker = MetricsTracker()
    tracker.load_trajectories(trajectories_dir)
    tracker.add_personas(personas)
    metrics = tracker.calculate_all_metrics()
    tracker.save_to_file(metrics_file)
    
    print(f"✅ {mode} simulation complete. Metrics saved.")
    return metrics

async def main():
    parser = argparse.ArgumentParser(description="Run full IntelliT evaluation")
    parser.add_argument('--personas', type=int, default=10, help='Number of personas')
    parser.add_argument('--concurrent', type=int, default=2, help='Max concurrent simulations')
    args = parser.parse_args()
    
    CONFIG['n_personas'] = args.personas
    CONFIG['max_concurrent'] = args.concurrent
    
    start_time = datetime.utcnow()
    init_logger(log_dir=f"{CONFIG['base_results_dir']}/logs")
    
    print_banner("INTELLIT FULL EVALUATION SUITE")
    print(f"Personas: {CONFIG['n_personas']}")
    print(f"Concurrent: {CONFIG['max_concurrent']}")
    
    # 1. Load Questions & Generate Personas
    print("\nInitializing...")
    questions = load_strivers_a2z_questions(limit=CONFIG['n_questions'])
    personas = generate_diverse_personas(n=CONFIG['n_personas'], seed=CONFIG['seed'])
    create_synthetic_users(personas) # Ensure they exist in DB
    
    # 2. Run Adaptive Simulation
    adaptive_dir = os.path.join(CONFIG['base_results_dir'], "adaptive")
    adaptive_metrics = await run_simulation_phase("adaptive", personas, questions, adaptive_dir)
    
    # 3. Run Baseline Simulation
    # Re-generate personas to reset state but keep same attributes/seed
    print("\nResetting personas for baseline...")
    personas_baseline = generate_diverse_personas(n=CONFIG['n_personas'], seed=CONFIG['seed'])
    # Note: We reuse the same DB users, but simulation tracks state internally in trajectory files
    # Ideally, we should reset DB state, but for simulation metrics (which are computed from
    # internal state updates in memory/files), this is sufficient as long as simulation_engine
    # doesn't pull stale state from DB at start of each day.
    # The simulation engine initializes DailyMetrics from current_mastery which is in the persona object.
    
    baseline_dir = os.path.join(CONFIG['base_results_dir'], "baseline")
    baseline_metrics = await run_simulation_phase("baseline", personas_baseline, questions, baseline_dir)
    
    # 4. Compare
    print_banner("EVALUATION SUMMARY")
    
    def get_metric(metrics, path):
        val = metrics
        for key in path.split('.'):
            val = val.get(key, {})
        return val if isinstance(val, (int, float)) else 0

    print(f"{'Metric':<30} | {'Adaptive':<15} | {'Baseline':<15} | {'Diff':<15}")
    print("-" * 80)
    
    comparisons = [
        ("Mastery Gain (Mean)", "online.learning_gains.mean"),
        ("Brier Score", "offline.mastery_calibration.brier_score"),
        ("ECE Score", "offline.mastery_calibration.ece_score"),
        ("AUROC", "offline.mastery_calibration.auroc"),
        ("Success Rate", "online.engagement.mean_success_rate"),
        ("Time to Mastery (Days)", "online.time_to_mastery.mean_days")
    ]
    
    for label, key in comparisons:
        a_val = get_metric(adaptive_metrics, key)
        b_val = get_metric(baseline_metrics, key)
        diff = a_val - b_val
        print(f"{label:<30} | {a_val:<15.4f} | {b_val:<15.4f} | {diff:<15.4f}")
    
    print("\nFull results saved to:", CONFIG['base_results_dir'])
    finalize_logger()

if __name__ == "__main__":
    asyncio.run(main())
