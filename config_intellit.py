"""
Configuration file for IntelliT Synthetic User Evaluation.

This file contains all configurable parameters for the simulation system.
Edit values here to customize the evaluation.
"""

import os

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Number of synthetic personas to generate
N_PERSONAS = 25

# Distribution of skill levels (must sum to 1.0)
SKILL_DISTRIBUTION = {
    'beginner': 0.4,      # 40%
    'intermediate': 0.4,  # 40%
    'advanced': 0.2       # 20%
}

# Number of questions to load from roadmap
N_QUESTIONS = 200

# Simulation duration (days)
SIMULATION_DAYS = 14

# Random seed for reproducibility (set to None for random)
RANDOM_SEED = 42


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

# Enable real IntelliT agent interactions
ENABLE_AGENTS = True

# Agent interaction settings
AGENT_CONFIG = {
    'enable_hints': True,           # Request hints when struggling
    'enable_code_analysis': True,   # Get code analysis on success
    'max_hints_per_problem': 3,     # Maximum hints to request per problem
    'hint_escalation': True,        # Escalate hint levels progressively
}


# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Maximum concurrent simulations (adjust based on system resources)
MAX_CONCURRENT_SIMULATIONS = 3

# Delay between operations (seconds) - prevents rate limiting
OPERATION_DELAY = 0.1


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# Database connection settings (inherited from backend/.env)
# Ensure backend is configured with proper ARANGO credentials

# Flag to mark synthetic users for easy cleanup
SYNTHETIC_USER_FLAG = True


# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Base results directory
RESULTS_DIR = "results"

# Subdirectories
TRAJECTORIES_DIR = os.path.join(RESULTS_DIR, "trajectories")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Output files
METRICS_FILE = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
CONTROL_METRICS_FILE = os.path.join(RESULTS_DIR, "control_metrics.json")
SUMMARY_REPORT = os.path.join(RESULTS_DIR, "summary_report.txt")


# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Run ablation study (compare with and without agents)
RUN_ABLATION = False

# Generate figures for paper
GENERATE_FIGURES = False

# Metrics to calculate
METRICS_ENABLED = {
    'offline': True,       # Mastery calibration, content coverage
    'online': True,        # Learning gains, time-to-mastery
    'agent': True,         # Hint effectiveness, code analysis
    'fairness': True,      # Learning gains by proficiency band
}


# ============================================================================
# ROADMAP FILTERING
# ============================================================================

# Roadmap course to load questions from
ROADMAP_COURSE = "strivers-a2z"

# Difficulty distribution preferences (None = use all)
DIFFICULTY_FILTER = None  # or ['Easy', 'Medium', 'Hard']

# Topic filters (None = use all topics)
TOPIC_FILTER = None  # or ['arrays', 'strings', 'trees']


# ============================================================================
# PERSONA GENERATION PARAMETERS
# ============================================================================

# Cognitive parameter ranges
COGNITIVE_PARAMS = {
    'beginner': {
        'learning_rate': (0.08, 0.12),
        'working_memory': (0.5, 0.7),
        'metacognition': (0.3, 0.5),
        'consistency': (0.6, 0.8),
        'hint_reliance': (0.65, 0.85),
    },
    'intermediate': {
        'learning_rate': (0.12, 0.18),
        'working_memory': (0.65, 0.85),
        'metacognition': (0.5, 0.75),
        'consistency': (0.75, 0.90),
        'hint_reliance': (0.35, 0.55),
    },
    'advanced': {
        'learning_rate': (0.05, 0.10),
        'working_memory': (0.80, 1.0),
        'metacognition': (0.75, 1.0),
        'consistency': (0.85, 0.95),
        'hint_reliance': (0.15, 0.30),
    }
}


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # Check skill distribution sums to 1.0
    if abs(sum(SKILL_DISTRIBUTION.values()) - 1.0) > 0.01:
        errors.append("SKILL_DISTRIBUTION must sum to 1.0")
    
    # Check positive values
    if N_PERSONAS <= 0:
        errors.append("N_PERSONAS must be positive")
    
    if N_QUESTIONS <= 0:
        errors.append("N_QUESTIONS must be positive")
    
    if SIMULATION_DAYS <= 0:
        errors.append("SIMULATION_DAYS must be positive")
    
    if MAX_CONCURRENT_SIMULATIONS <= 0:
        errors.append("MAX_CONCURRENT_SIMULATIONS must be positive")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


# Validate on import
validate_config()

