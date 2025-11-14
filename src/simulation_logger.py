"""
Simulation Logger - Centralized logging for debugging and analysis.

Captures all simulation events to file while showing clean console output.
DRY principle: Single source of truth for logging.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class SimulationLogger:
    """
    Centralized logger for simulation events.
    
    Logs to both console (clean) and file (detailed) simultaneously.
    """
    
    def __init__(self, log_dir: str = "student_sim/results/logs"):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"simulation_{timestamp}.log"
        self.json_log_file = self.log_dir / f"simulation_{timestamp}.jsonl"
        
        # Session metadata
        self.session_start = datetime.now()
        self.events = []
        
        self._write_header()
    
    def _write_header(self):
        """Write log file header."""
        with open(self.log_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"IntelliT Synthetic User Simulation Log\n")
            f.write(f"Started: {self.session_start.isoformat()}\n")
            f.write(f"{'='*80}\n\n")
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """
        Convert numpy types and other non-JSON-serializable objects to native Python types.
        
        Args:
            obj: Object to sanitize
        
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def log_event(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        console: bool = True,
        console_prefix: str = ""
    ):
        """
        Log an event to file and optionally console.
        
        Args:
            event_type: Type of event (e.g., 'cognitive_prediction', 'llm_call', 'submission')
            message: Human-readable message
            data: Structured data for debugging
            console: Whether to print to console
            console_prefix: Prefix for console output (e.g., "  ")
        """
        timestamp = datetime.now()
        
        # Sanitize data for JSON serialization (handles numpy types)
        sanitized_data = self._sanitize_for_json(data) if data else {}
        
        # Create event record
        event = {
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "message": message,
            "data": sanitized_data
        }
        
        self.events.append(event)
        
        # Write to text log
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] [{event_type}] {message}\n")
            if sanitized_data:
                f.write(f"  Data: {json.dumps(sanitized_data, indent=2)}\n")
            f.write("\n")
        
        # Write to JSON log (for structured analysis)
        with open(self.json_log_file, 'a') as f:
            f.write(json.dumps(event) + "\n")
        
        # Console output (clean)
        if console:
            print(f"{console_prefix}{message}")
    
    def log_llm_call(
        self,
        model: str,
        prompt_length: int,
        response_length: int,
        purpose: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log LLM API call details."""
        self.log_event(
            event_type="llm_call",
            message=f"🤖 LLM Call: {purpose} ({prompt_length} → {response_length} chars)",
            data={
                "model": model,
                "prompt_length": prompt_length,
                "response_length": response_length,
                "purpose": purpose,
                **(metadata or {})
            },
            console=True,
            console_prefix="  "
        )
    
    def log_cognitive_prediction(
        self,
        persona_key: str,
        problem_title: str,
        attempt: int,
        prediction: Dict[str, Any]
    ):
        """Log cognitive error prediction."""
        will_error = prediction['will_make_error']
        error_prob = prediction['error_probability']
        mastery = prediction['avg_topic_mastery']
        
        self.log_event(
            event_type="cognitive_prediction",
            message=f"🧠 Prediction: {'ERROR' if will_error else 'SUCCESS'} ({error_prob:.1%} prob, {mastery:.1%} mastery)",
            data={
                "persona": persona_key,
                "problem": problem_title,
                "attempt": attempt,
                **prediction
            },
            console=True,
            console_prefix="  "
        )
    
    def log_submission(
        self,
        persona_key: str,
        problem_title: str,
        attempt: int,
        success: bool,
        passed: int,
        total: int,
        error_message: Optional[str] = None
    ):
        """Log code submission result."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.log_event(
            event_type="submission",
            message=f"📊 Submission: {status} ({passed}/{total} tests)",
            data={
                "persona": persona_key,
                "problem": problem_title,
                "attempt": attempt,
                "success": success,
                "passed_count": passed,
                "total_count": total,
                "error_message": error_message
            },
            console=True,
            console_prefix="  "
        )
    
    def log_agent_interaction(
        self,
        agent_type: str,
        persona_key: str,
        action: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log interaction with IntelliT agents."""
        icon = {"hint": "💡", "analysis": "🔍", "orchestrator": "💬"}.get(agent_type, "🤝")
        status = "Success" if success else "Failed"
        
        self.log_event(
            event_type=f"agent_{agent_type}",
            message=f"{icon} {agent_type.title()}: {action} - {status}",
            data={
                "persona": persona_key,
                "agent": agent_type,
                "action": action,
                "success": success,
                **(metadata or {})
            },
            console=True,
            console_prefix="  "
        )
    
    def log_day_summary(
        self,
        day: int,
        persona_key: str,
        problem_title: str,
        attempts: int,
        success: bool,
        hints_used: int,
        mastery_change: float
    ):
        """Log daily summary."""
        status = "✅" if success else "❌"
        self.log_event(
            event_type="day_summary",
            message=f"Day {day}: {status} {problem_title} ({attempts} attempts, {hints_used} hints, Δmastery: {mastery_change:+.3f})",
            data={
                "day": day,
                "persona": persona_key,
                "problem": problem_title,
                "attempts": attempts,
                "success": success,
                "hints_used": hints_used,
                "mastery_change": mastery_change
            },
            console=True,
            console_prefix=""
        )
    
    def log_simulation_complete(
        self,
        persona_key: str,
        days_active: int,
        total_attempts: int,
        success_rate: float,
        mastery_gain: float,
        total_hints: int
    ):
        """Log simulation completion summary."""
        self.log_event(
            event_type="simulation_complete",
            message=f"✅ Complete: {persona_key} | {days_active} days | {success_rate:.1%} success | +{mastery_gain:.3f} mastery | {total_hints} hints",
            data={
                "persona": persona_key,
                "days_active": days_active,
                "total_attempts": total_attempts,
                "success_rate": success_rate,
                "mastery_gain": mastery_gain,
                "total_hints": total_hints,
                "duration_seconds": (datetime.now() - self.session_start).total_seconds()
            },
            console=True
        )
    
    def finalize(self):
        """Finalize log file with summary."""
        duration = (datetime.now() - self.session_start).total_seconds()
        
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Simulation Complete\n")
            f.write(f"Duration: {duration:.1f}s\n")
            f.write(f"Total Events: {len(self.events)}\n")
            f.write(f"{'='*80}\n")
        
        print(f"\n📝 Full logs saved to: {self.log_file}")
        print(f"📊 Structured logs: {self.json_log_file}")


# Global logger instance
_logger: Optional[SimulationLogger] = None


def get_logger() -> SimulationLogger:
    """Get global logger instance."""
    global _logger
    if _logger is None:
        _logger = SimulationLogger()
    return _logger


def init_logger(log_dir: str = "student_sim/results/logs") -> SimulationLogger:
    """Initialize global logger."""
    global _logger
    _logger = SimulationLogger(log_dir)
    return _logger


def finalize_logger():
    """Finalize and close logger."""
    global _logger
    if _logger:
        _logger.finalize()
        _logger = None

