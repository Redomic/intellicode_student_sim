"""
IntelliT Synthetic User Evaluation - Main Runner

This script orchestrates the complete synthetic user evaluation pipeline:
1. Load questions from strivers-a2z roadmap
2. Generate diverse synthetic personas
3. Create database users
4. Run 14-day simulations with real IntelliT agents
5. Calculate research-grade evaluation metrics
6. Generate figures for paper

Usage:
    conda activate intellicode
    cd student_sim
    python main_intellit.py

Configuration:
    Edit config below or set environment variables
"""

import sys
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '../backend'))

from src.intellit_adapter import load_strivers_a2z_questions
from src.persona_generator import generate_diverse_personas, print_persona_summary
from src.db_setup import (
    create_synthetic_users,
    verify_synthetic_users,
    get_synthetic_user_count
)
from src.simulation_engine import run_parallel_simulations
from src.metrics_tracker import MetricsTracker


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Simulation parameters
    "n_personas": 25,  # Number of synthetic users
    "n_questions": 200,  # Number of questions to load
    "max_concurrent": 3,  # Max parallel simulations (API rate limiting)
    "random_seed": 42,  # For reproducibility
    
    # Output paths
    "results_dir": "results",
    "trajectories_dir": "results/trajectories",
    "metrics_file": "results/evaluation_metrics.json",
    
    # Ablation study (not implemented - requires separate control backend)
    "run_ablation": False
}


def print_banner(text: str):
    """Print formatted banner."""
    print(f"\n{'='*80}")
    print(f"{text:^80}")
    print(f"{'='*80}\n")


async def main():
    """Main execution pipeline."""
    start_time = datetime.utcnow()
    
    print_banner("INTELLIT SYNTHETIC USER EVALUATION")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: {CONFIG['n_personas']} personas, {CONFIG['n_questions']} questions")
    print(f"Backend API: http://localhost:8000")
    print(f"Max Concurrent: {CONFIG['max_concurrent']}")
    
    # ========================================================================
    # PRE-FLIGHT CHECKS: Fail fast if requirements not met
    # ========================================================================
    print("\n🔍 Running pre-flight checks...")
    
    # Check GEMINI_API_KEY
    if not os.getenv('GEMINI_API_KEY'):
        print("\n❌ FATAL ERROR: GEMINI_API_KEY environment variable not set")
        print("⛔ Set it in .env file or export GEMINI_API_KEY=your_key")
        return 1
    
    # Check backend is running
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/docs", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    print(f"\n❌ FATAL ERROR: Backend returned status {response.status}")
                    print("⛔ Start backend with: cd backend && uvicorn app.main:app --reload")
                    return 1
    except Exception as e:
        print(f"\n❌ FATAL ERROR: Cannot connect to backend at http://localhost:8000")
        print(f"Error: {e}")
        print("⛔ Start backend with: cd backend && uvicorn app.main:app --reload")
        return 1
    
    print("✅ GEMINI_API_KEY configured")
    print("✅ Backend is running")
    
    # Create output directories
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    os.makedirs(CONFIG['trajectories_dir'], exist_ok=True)
    
    # ========================================================================
    # PHASE 1: Load Questions from Roadmap
    # ========================================================================
    
    print_banner("PHASE 1: LOADING QUESTIONS")
    
    try:
        questions = load_strivers_a2z_questions(limit=CONFIG['n_questions'])
        
        if not questions:
            print("❌ No questions loaded. Aborting.")
            return 1
        
        # Print question statistics
        difficulties = {}
        for q in questions:
            diff = q.get('difficulty', 'Unknown')
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        print(f"\n✅ Loaded {len(questions)} questions")
        print(f"   Difficulty Distribution:")
        for diff, count in sorted(difficulties.items()):
            print(f"     - {diff}: {count}")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: Failed to load questions")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n⛔ ABORTING SIMULATION - Fix errors before proceeding")
        return 1
    
    # ========================================================================
    # PHASE 2: Generate Synthetic Personas
    # ========================================================================
    
    print_banner("PHASE 2: GENERATING PERSONAS")
    
    try:
        personas = generate_diverse_personas(
            n=CONFIG['n_personas'],
            seed=CONFIG['random_seed']
        )
        
        # Print sample personas
        print("\nSample Personas:")
        for i in [0, len(personas)//2, -1]:  # Beginning, middle, end
            print_persona_summary(personas[i])
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: Failed to generate personas")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n⛔ ABORTING SIMULATION - Fix errors before proceeding")
        return 1
    
    # ========================================================================
    # PHASE 3: Create Database Users
    # ========================================================================
    
    print_banner("PHASE 3: CREATING DATABASE USERS")
    
    try:
        # Check existing count
        existing_count = get_synthetic_user_count()
        print(f"ℹ️ Existing synthetic users in database: {existing_count}")
        
        # Create users
        user_keys = create_synthetic_users(personas)
        
        # Verify
        verification = verify_synthetic_users(personas)
        
        if verification['users_missing'] > 0:
            print(f"⚠️ Warning: {verification['users_missing']} users could not be verified")
            if verification['missing_keys']:
                print(f"   Missing: {verification['missing_keys']}")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: Failed to create database users")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n⛔ ABORTING SIMULATION - Fix errors before proceeding")
        return 1
    
    # ========================================================================
    # PHASE 4: Run Simulations
    # ========================================================================
    
    print_banner("PHASE 4: RUNNING SIMULATIONS")
    
    try:
        print(f"Starting {CONFIG['n_personas']} parallel simulations...")
        print(f"Estimated time: 5-10 minutes\n")
        
        # Run simulations (uses backend API, agents always enabled)
        results = await run_parallel_simulations(
            personas=personas,
            questions=questions,
            max_concurrent=CONFIG['max_concurrent']
        )
        
        print(f"\n✅ Completed {len(results)} simulations")
        
        # Print summary statistics
        total_attempts = sum(len(traj) for traj in results.values())
        print(f"   Total Problem Attempts: {total_attempts}")
        
        successful_days = sum(
            sum(1 for m in traj if m.success)
            for traj in results.values()
        )
        print(f"   Successful Attempts: {successful_days}")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: Failed to run simulations")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n⛔ ABORTING SIMULATION - Fix errors before proceeding")
        return 1
    
    # ========================================================================
    # PHASE 5: Calculate Metrics
    # ========================================================================
    
    print_banner("PHASE 5: CALCULATING METRICS")
    
    try:
        # Initialize metrics tracker
        tracker = MetricsTracker()
        
        # Load trajectories from files
        tracker.load_trajectories(CONFIG['trajectories_dir'])
        
        # Add persona data
        tracker.add_personas(personas)
        
        # Calculate all metrics
        metrics = tracker.calculate_all_metrics()
        
        # Save to file
        tracker.save_to_file(CONFIG['metrics_file'])
        
        # Print summary
        tracker.print_summary()
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: Failed to calculate metrics")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n⛔ ABORTING SIMULATION - Fix errors before proceeding")
        return 1
    
    # ========================================================================
    # PHASE 6: Ablation Study (Optional)
    # ========================================================================
    
    if CONFIG['run_ablation']:
        print_banner("PHASE 6: ABLATION STUDY (NOT IMPLEMENTED)")
        print("⚠️ Ablation study requires separate backend configuration.")
        print("   All simulations use the backend API which includes agent interactions.")
        print("   To run ablation, you would need to disable agents in backend config.")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    
    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()
    
    print_banner("SIMULATION COMPLETE")
    
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    print(f"\n📊 Results:")
    print(f"   - Trajectories: {CONFIG['trajectories_dir']}/")
    print(f"   - Metrics: {CONFIG['metrics_file']}")
    
    print(f"\n{'='*80}")
    print("✅ ALL PHASES COMPLETE")
    print(f"{'='*80}\n")
    
    return 0


def cli():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run IntelliT synthetic user evaluation"
    )
    parser.add_argument(
        '--personas', '-n',
        type=int,
        default=25,
        help='Number of synthetic personas (default: 25)'
    )
    parser.add_argument(
        '--questions', '-q',
        type=int,
        default=200,
        help='Number of questions to load (default: 200)'
    )
    parser.add_argument(
        '--concurrent', '-c',
        type=int,
        default=3,
        help='Max concurrent simulations (default: 3)'
    )
    parser.add_argument(
        '--ablation',
        action='store_true',
        help='Run ablation study (with and without agents)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Update config from args
    CONFIG['n_personas'] = args.personas
    CONFIG['n_questions'] = args.questions
    CONFIG['max_concurrent'] = args.concurrent
    CONFIG['run_ablation'] = args.ablation
    CONFIG['random_seed'] = args.seed
    
    # Run main
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ INTERRUPTED BY USER")
        print("⛔ Simulation stopped - partial data may be incomplete")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR IN MAIN: {e}")
        import traceback
        traceback.print_exc()
        print("\n⛔ SIMULATION FAILED - No valid data generated")
        sys.exit(1)


if __name__ == "__main__":
    cli()

