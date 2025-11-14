"""
Metrics Tracker - Calculate research-grade evaluation metrics.

Generates metrics for the conference demo paper:
1. Offline Metrics: Mastery calibration, content coverage
2. Online Metrics: Learning gains, time-to-mastery
3. Agent Metrics: Hint effectiveness, code analysis adoption
4. Fairness Metrics: Learning gains by proficiency band

All metrics are calculated from real data - NO PLACEHOLDERS.
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter

from .persona_generator import SyntheticPersona, get_persona_average_mastery
from .simulation_engine import DailyMetrics


class MetricsTracker:
    """
    Calculate comprehensive evaluation metrics from simulation trajectories.
    """
    
    def __init__(self):
        self.metrics = {}
        self.trajectories = {}
        self.personas = {}
    
    def load_trajectories(self, trajectory_dir: str = "student_sim/results/trajectories"):
        """
        Load all trajectory files from directory.
        
        Args:
            trajectory_dir: Directory containing trajectory JSON files
        """
        if not os.path.exists(trajectory_dir):
            print(f"⚠️ Trajectory directory not found: {trajectory_dir}")
            return
        
        count = 0
        for filename in os.listdir(trajectory_dir):
            if filename.endswith("_trajectory.json"):
                filepath = os.path.join(trajectory_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        persona_key = data['persona_key']
                        self.trajectories[persona_key] = data
                        count += 1
                except Exception as e:
                    print(f"⚠️ Error loading {filename}: {e}")
        
        print(f"✅ Loaded {count} trajectories")
    
    def add_personas(self, personas: List[SyntheticPersona]):
        """Add persona objects for reference."""
        for persona in personas:
            self.personas[persona.user_key] = persona
    
    def calculate_all_metrics(self):
        """Calculate all evaluation metrics."""
        print(f"\n{'='*60}")
        print("Calculating Evaluation Metrics")
        print(f"{'='*60}\n")
        
        self.metrics = {
            "offline": self._calculate_offline_metrics(),
            "online": self._calculate_online_metrics(),
            "agent": self._calculate_agent_metrics(),
            "fairness": self._calculate_fairness_metrics(),
            "metadata": {
                "calculation_time": datetime.utcnow().isoformat(),
                "total_trajectories": len(self.trajectories),
                "total_personas": len(self.personas)
            }
        }
        
        print(f"\n{'='*60}")
        print("✅ Metrics Calculation Complete")
        print(f"{'='*60}\n")
        
        return self.metrics
    
    def _calculate_offline_metrics(self) -> dict:
        """Calculate offline metrics (mastery calibration, content coverage)."""
        print("📊 Calculating Offline Metrics...")
        
        # Mastery calibration: predicted vs actual success
        predicted_successes = []
        actual_successes = []
        
        for persona_key, trajectory_data in self.trajectories.items():
            for day_metrics in trajectory_data['daily_metrics']:
                if not day_metrics['active']:
                    continue
                
                # Use mastery_before as predictor
                mastery_before = day_metrics.get('mastery_before', {})
                topics = day_metrics.get('problem_topics', [])
                
                if topics and mastery_before:
                    avg_mastery = np.mean([mastery_before.get(t, 0.5) for t in topics])
                    predicted_successes.append(avg_mastery)
                    actual_successes.append(1.0 if day_metrics['success'] else 0.0)
        
        # Calculate Brier score
        brier_score = 0.0
        if predicted_successes:
            brier_score = np.mean([(p - a)**2 for p, a in zip(predicted_successes, actual_successes)])
        
        # Calculate correlation
        correlation = 0.0
        if len(predicted_successes) > 1:
            correlation = np.corrcoef(predicted_successes, actual_successes)[0, 1]
        
        # Content coverage: topic distribution
        topic_attempts = Counter()
        for trajectory_data in self.trajectories.values():
            for day_metrics in trajectory_data['daily_metrics']:
                if day_metrics['active']:
                    topics = day_metrics.get('problem_topics', [])
                    for topic in topics:
                        topic_attempts[topic] += 1
        
        # Calculate coverage uniformity (lower = more uniform)
        if topic_attempts:
            topic_counts = list(topic_attempts.values())
            coverage_cv = np.std(topic_counts) / np.mean(topic_counts) if np.mean(topic_counts) > 0 else 0
        else:
            coverage_cv = 0
        
        offline_metrics = {
            "mastery_calibration": {
                "brier_score": round(brier_score, 4),
                "correlation": round(correlation, 4),
                "n_predictions": len(predicted_successes)
            },
            "content_coverage": {
                "unique_topics_covered": len(topic_attempts),
                "coverage_coefficient_of_variation": round(coverage_cv, 4),
                "topic_distribution": dict(topic_attempts.most_common(10))
            }
        }
        
        print("  ✅ Mastery calibration calculated")
        print("  ✅ Content coverage calculated")
        
        return offline_metrics
    
    def _calculate_online_metrics(self) -> dict:
        """Calculate online metrics (learning gains, time-to-mastery)."""
        print("📊 Calculating Online Metrics...")
        
        learning_gains = []
        time_to_mastery = []
        learning_velocities = []
        
        for persona_key, trajectory_data in self.trajectories.items():
            initial_mastery = trajectory_data.get('initial_avg_mastery', 0)
            final_mastery = trajectory_data.get('final_avg_mastery', 0)
            gain = final_mastery - initial_mastery
            learning_gains.append(gain)
            
            # Calculate time-to-mastery (days to reach 0.7)
            days_to_07 = None
            for day_metrics in trajectory_data['daily_metrics']:
                if day_metrics['active']:
                    mastery_after = day_metrics.get('mastery_after', {})
                    if mastery_after:
                        avg_mastery = np.mean(list(mastery_after.values()))
                        if avg_mastery >= 0.7 and days_to_07 is None:
                            days_to_07 = day_metrics['day']
                            break
            
            if days_to_07:
                time_to_mastery.append(days_to_07)
            
            # Calculate learning velocity (mastery gain per active day)
            active_days = trajectory_data.get('total_days', 14)
            if active_days > 0:
                velocity = gain / active_days
                learning_velocities.append(velocity)
        
        # Engagement metrics
        total_active_days = []
        success_rates = []
        hints_per_problem = []
        
        for trajectory_data in self.trajectories.values():
            total_active_days.append(trajectory_data.get('total_days', 0))
            
            attempts = trajectory_data.get('total_attempts', 0)
            successes = trajectory_data.get('total_successes', 0)
            if attempts > 0:
                success_rates.append(successes / attempts)
            
            # Calculate hints per problem
            total_hints = sum(
                m.get('hints_used', 0) 
                for m in trajectory_data['daily_metrics']
                if m.get('active')
            )
            active_days_count = len([m for m in trajectory_data['daily_metrics'] if m.get('active')])
            if active_days_count > 0:
                hints_per_problem.append(total_hints / active_days_count)
        
        online_metrics = {
            "learning_gains": {
                "mean": round(np.mean(learning_gains), 4) if learning_gains else 0,
                "std": round(np.std(learning_gains), 4) if learning_gains else 0,
                "median": round(np.median(learning_gains), 4) if learning_gains else 0,
                "min": round(np.min(learning_gains), 4) if learning_gains else 0,
                "max": round(np.max(learning_gains), 4) if learning_gains else 0,
                "n": len(learning_gains)
            },
            "time_to_mastery": {
                "mean_days": round(np.mean(time_to_mastery), 2) if time_to_mastery else None,
                "median_days": round(np.median(time_to_mastery), 2) if time_to_mastery else None,
                "n_reached_mastery": len(time_to_mastery),
                "pct_reached_mastery": round(len(time_to_mastery) / len(self.trajectories) * 100, 1) if self.trajectories else 0
            },
            "learning_velocity": {
                "mean_gain_per_day": round(np.mean(learning_velocities), 4) if learning_velocities else 0,
                "median_gain_per_day": round(np.median(learning_velocities), 4) if learning_velocities else 0
            },
            "engagement": {
                "mean_active_days": round(np.mean(total_active_days), 2) if total_active_days else 0,
                "mean_success_rate": round(np.mean(success_rates), 4) if success_rates else 0,
                "mean_hints_per_problem": round(np.mean(hints_per_problem), 2) if hints_per_problem else 0
            }
        }
        
        print("  ✅ Learning gains calculated")
        print("  ✅ Time-to-mastery calculated")
        print("  ✅ Engagement metrics calculated")
        
        return online_metrics
    
    def _calculate_agent_metrics(self) -> dict:
        """Calculate agent effectiveness metrics."""
        print("📊 Calculating Agent Metrics...")
        
        hint_interactions = []
        code_analyses = []
        
        for trajectory_data in self.trajectories.values():
            for day_metrics in trajectory_data['daily_metrics']:
                if not day_metrics['active']:
                    continue
                
                # Hint effectiveness
                hints_used = day_metrics.get('hints_used', 0)
                if hints_used > 0:
                    hint_interactions.append({
                        'hints_count': hints_used,
                        'success': day_metrics.get('success', False)
                    })
                
                # Code analysis
                analysis_suggestions = day_metrics.get('code_analysis_suggestions', 0)
                if analysis_suggestions > 0:
                    code_analyses.append({
                        'suggestion_count': analysis_suggestions,
                        'success': day_metrics.get('success', False)
                    })
        
        # Calculate hint effectiveness
        hint_success_rate = 0.0
        if hint_interactions:
            successful_with_hints = sum(1 for h in hint_interactions if h['success'])
            hint_success_rate = successful_with_hints / len(hint_interactions)
        
        agent_metrics = {
            "hint_effectiveness": {
                "total_hint_interactions": len(hint_interactions),
                "success_rate_with_hints": round(hint_success_rate, 4),
                "mean_hints_per_interaction": round(np.mean([h['hints_count'] for h in hint_interactions]), 2) if hint_interactions else 0
            },
            "code_analysis": {
                "total_analyses": len(code_analyses),
                "mean_suggestions_per_analysis": round(np.mean([c['suggestion_count'] for c in code_analyses]), 2) if code_analyses else 0,
                "analyses_on_successful_submissions": len(code_analyses)
            }
        }
        
        print("  ✅ Hint effectiveness calculated")
        print("  ✅ Code analysis metrics calculated")
        
        return agent_metrics
    
    def _calculate_fairness_metrics(self) -> dict:
        """Calculate fairness metrics across proficiency bands."""
        print("📊 Calculating Fairness Metrics...")
        
        # Group by skill level
        by_skill = defaultdict(list)
        
        for persona_key, trajectory_data in self.trajectories.items():
            skill_level = trajectory_data.get('skill_level', 'unknown')
            learning_gain = trajectory_data.get('mastery_gain', 0)
            by_skill[skill_level].append(learning_gain)
        
        fairness_metrics = {
            "learning_gains_by_skill": {},
            "fairness_analysis": {}
        }
        
        for skill_level, gains in by_skill.items():
            fairness_metrics["learning_gains_by_skill"][skill_level] = {
                "mean_gain": round(np.mean(gains), 4),
                "std_gain": round(np.std(gains), 4),
                "median_gain": round(np.median(gains), 4),
                "n": len(gains)
            }
        
        # Calculate inter-quartile range check
        all_gains = []
        for gains in by_skill.values():
            all_gains.extend(gains)
        
        if all_gains:
            q1 = np.percentile(all_gains, 25)
            q3 = np.percentile(all_gains, 75)
            iqr = q3 - q1
            median_gain = np.median(all_gains)
            
            fairness_metrics["fairness_analysis"] = {
                "overall_median_gain": round(median_gain, 4),
                "iqr": round(iqr, 4),
                "iqr_pct_of_median": round((iqr / median_gain * 100) if median_gain != 0 else 0, 2),
                "fairness_target_met": (iqr / median_gain <= 0.15) if median_gain != 0 else False
            }
        
        print("  ✅ Fairness metrics calculated")
        
        return fairness_metrics
    
    def save_to_file(self, filepath: str = "student_sim/results/evaluation_metrics.json"):
        """
        Save all metrics to JSON file.
        
        Args:
            filepath: Output file path
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            print(f"\n✅ Metrics saved to {filepath}")
            
        except Exception as e:
            print(f"\n❌ Error saving metrics: {e}")
    
    def print_summary(self):
        """Print human-readable summary of metrics."""
        if not self.metrics:
            print("⚠️ No metrics calculated yet. Call calculate_all_metrics() first.")
            return
        
        print(f"\n{'='*60}")
        print("EVALUATION METRICS SUMMARY")
        print(f"{'='*60}\n")
        
        # Offline metrics
        print("📊 OFFLINE METRICS")
        print("-" * 60)
        offline = self.metrics.get('offline', {})
        calib = offline.get('mastery_calibration', {})
        print(f"  Brier Score: {calib.get('brier_score', 0):.4f}")
        print(f"  Correlation: {calib.get('correlation', 0):.4f}")
        print(f"  Predictions: {calib.get('n_predictions', 0)}")
        
        coverage = offline.get('content_coverage', {})
        print(f"  Topics Covered: {coverage.get('unique_topics_covered', 0)}")
        print(f"  Coverage CV: {coverage.get('coverage_coefficient_of_variation', 0):.4f}")
        
        # Online metrics
        print(f"\n📈 ONLINE METRICS")
        print("-" * 60)
        online = self.metrics.get('online', {})
        gains = online.get('learning_gains', {})
        print(f"  Mean Learning Gain: {gains.get('mean', 0):.4f} ± {gains.get('std', 0):.4f}")
        print(f"  Median Learning Gain: {gains.get('median', 0):.4f}")
        print(f"  Range: [{gains.get('min', 0):.4f}, {gains.get('max', 0):.4f}]")
        
        ttm = online.get('time_to_mastery', {})
        print(f"  Mean Time to Mastery: {ttm.get('mean_days', 'N/A')} days")
        print(f"  % Reached Mastery: {ttm.get('pct_reached_mastery', 0):.1f}%")
        
        velocity = online.get('learning_velocity', {})
        print(f"  Learning Velocity: {velocity.get('mean_gain_per_day', 0):.4f} gain/day")
        
        # Agent metrics
        print(f"\n🤖 AGENT METRICS")
        print("-" * 60)
        agent = self.metrics.get('agent', {})
        hints = agent.get('hint_effectiveness', {})
        print(f"  Hint Interactions: {hints.get('total_hint_interactions', 0)}")
        print(f"  Success Rate with Hints: {hints.get('success_rate_with_hints', 0):.2%}")
        print(f"  Avg Hints per Problem: {hints.get('mean_hints_per_interaction', 0):.2f}")
        
        analysis = agent.get('code_analysis', {})
        print(f"  Code Analyses: {analysis.get('total_analyses', 0)}")
        print(f"  Avg Suggestions: {analysis.get('mean_suggestions_per_analysis', 0):.2f}")
        
        # Fairness metrics
        print(f"\n⚖️  FAIRNESS METRICS")
        print("-" * 60)
        fairness = self.metrics.get('fairness', {})
        by_skill = fairness.get('learning_gains_by_skill', {})
        for skill, stats in by_skill.items():
            print(f"  {skill.title()}: {stats.get('mean_gain', 0):.4f} ± {stats.get('std_gain', 0):.4f} (n={stats.get('n', 0)})")
        
        fair_analysis = fairness.get('fairness_analysis', {})
        print(f"  IQR: {fair_analysis.get('iqr', 0):.4f}")
        print(f"  IQR % of Median: {fair_analysis.get('iqr_pct_of_median', 0):.2f}%")
        print(f"  Fairness Target Met: {'✅' if fair_analysis.get('fairness_target_met') else '❌'}")
        
        print(f"\n{'='*60}\n")

