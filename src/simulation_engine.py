"""
Simulation Engine - Orchestrate 14-day accelerated learning trajectories.

This module combines:
1. Student_sim's cognitive simulation (realistic mistake generation)
2. IntelliT backend API (code execution, evaluation, learner state)
3. Real multi-agent interactions via API (hints, feedback, analysis)
4. Mastery tracking and learning gains measurement

The simulation uses the backend API like a headless user, ensuring ALL
metrics are properly tracked (sessions, submissions, learner state).
"""

import asyncio
import random
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os

from .persona_generator import (
    SyntheticPersona, 
    should_request_hint,
    update_persona_mastery,
    get_persona_average_mastery
)
from .agent_integration import IntelliTAPIClient
from .cognitive_behavior import (
    predict_student_errors,
    update_cognitive_state,
    calculate_target_success_rate
)
from .error_injection import generate_code_with_errors
from .simulation_logger import get_logger


class DailyMetrics:
    """Track metrics for a single day of simulation."""
    
    def __init__(self, day: int, persona_key: str):
        self.day = day
        self.persona_key = persona_key
        self.timestamp = datetime.utcnow()
        
        # Activity metrics
        self.active = False
        self.skipped_reason = None
        
        # Problem metrics
        self.problem_attempted = None
        self.problem_id = None
        self.problem_difficulty = None
        self.problem_topics = []
        
        # Performance metrics
        self.attempts = 0
        self.hints_used = 0
        self.time_spent_seconds = 0
        self.success = False
        self.submission_id = None
        
        # Session metrics
        self.session_id = None
        
        # Mastery metrics
        self.mastery_before = {}
        self.mastery_after = {}
        self.mastery_delta = {}
        
        # Agent interaction metrics
        self.hints_received = []
        
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "day": self.day,
            "persona_key": self.persona_key,
            "timestamp": self.timestamp.isoformat(),
            "active": self.active,
            "skipped_reason": self.skipped_reason,
            "problem_attempted": self.problem_attempted,
            "problem_id": self.problem_id,
            "problem_difficulty": self.problem_difficulty,
            "problem_topics": self.problem_topics,
            "attempts": self.attempts,
            "hints_used": self.hints_used,
            "time_spent_seconds": self.time_spent_seconds,
            "success": self.success,
            "submission_id": self.submission_id,
            "session_id": self.session_id,
            "mastery_before": self.mastery_before,
            "mastery_after": self.mastery_after,
            "mastery_delta": self.mastery_delta,
            "hints_received_count": len(self.hints_received)
        }


def select_problem_for_persona(
    persona: SyntheticPersona,
    questions: List[Dict[str, Any]],
    day: int,
    attempted_questions: set
) -> Optional[Dict[str, Any]]:
    """
    Select appropriate problem for persona based on their mastery.
    
    Mimics Content Curator logic:
    - Growth zone: Pick problems with topics in mastery range [0.3, 0.7]
    - Challenge: Occasionally pick harder problems
    - Review: For high mastery topics
    
    Args:
        persona: SyntheticPersona
        questions: Available questions
        day: Current day number
        attempted_questions: Set of question IDs already attempted recently
    
    Returns:
        Selected question dict or None
    """
    if not questions:
        return None
    
    # Filter out recently attempted questions
    available_questions = [
        q for q in questions 
        if q.get('question_id') not in attempted_questions
    ]
    
    if not available_questions:
        # If all attempted, clear history and use all
        available_questions = questions
        attempted_questions.clear()
    
    # Calculate topic difficulties based on persona's mastery
    scored_questions = []
    
    for q in available_questions:
        topics = q.get('topics', [])
        if not topics:
            continue
        
        # Calculate average mastery for this question's topics
        topic_masteries = [
            persona.current_mastery.get(t, 0.5) 
            for t in topics
        ]
        avg_mastery = sum(topic_masteries) / len(topic_masteries) if topic_masteries else 0.5
        
        # Score based on growth zone preference
        if 0.3 <= avg_mastery <= 0.7:
            score = 1.0  # Optimal growth zone
        elif avg_mastery < 0.3:
            score = 0.7  # Challenge zone
        else:
            score = 0.5  # Review/consolidation
        
        # Add randomness
        score += random.uniform(-0.2, 0.2)
        
        scored_questions.append((q, score))
    
    if not scored_questions:
        return None
    
    # Sort by score
    scored_questions.sort(key=lambda x: x[1], reverse=True)
    
    # Pick from top candidates with some randomness
    top_n = min(10, len(scored_questions))
    selected = random.choice(scored_questions[:top_n])[0]
    
    return selected


async def generate_code_with_cognitive_modeling(
    persona: SyntheticPersona,
    problem: Dict[str, Any],
    attempt_number: int,
    previous_errors: Optional[str] = None,
    previous_code: Optional[str] = None,
    previous_hint: Optional[str] = None
) -> str:
    """
    Generate code using student_sim's cognitive modeling methodology.
    
    This implements the research-grade approach:
    1. Predict errors based on cognitive profile and topic mastery
    2. Generate code with realistic, intentional mistakes
    3. Show learning progression with hint incorporation
    
    Args:
        persona: SyntheticPersona with cognitive profile
        problem: Question dict with problem statement
        attempt_number: Current attempt number (1-indexed)
        previous_errors: Error message from previous attempt
        previous_code: Previous code attempt
        previous_hint: Hint text received from IntelliT agents
    
    Returns:
        Python code string with realistic student mistakes
    """
    
    # Step 1: Predict errors based on cognitive profile
    error_prediction = predict_student_errors(
        persona=persona,
        problem=problem,
        attempt_number=attempt_number,
        previous_hint=previous_hint
    )
    
    # Log cognitive prediction
    logger = get_logger()
    logger.log_cognitive_prediction(
        persona_key=persona.user_key,
        problem_title=problem.get('title', 'Untitled'),
        attempt=attempt_number,
        prediction=error_prediction
    )
    
    # Step 2: Generate code with predicted errors using error injection
    code = await generate_code_with_errors(
        persona=persona,
        problem=problem,
        error_prediction=error_prediction,
        attempt_number=attempt_number,
        previous_code=previous_code,
        previous_errors=previous_errors,
        previous_hint=previous_hint
    )
    
    return code


async def simulate_problem_attempt(
    api_client: IntelliTAPIClient,
    problem: Dict[str, Any],
    max_attempts: int = 3,
    use_hints: bool = True
) -> Dict[str, Any]:
    """
    Simulate attempting a single problem with multiple tries and hint requests.
    
    This is the core interaction loop:
    1. Start session for problem
    2. Generate code attempt
    3. Submit via API (automatic learner state update)
    4. If failed and allowed, request hint
    5. Repeat up to max_attempts
    6. End session
    
    Args:
        api_client: Authenticated IntelliTAPIClient
        problem: Question dict
        max_attempts: Maximum attempts allowed (default: 3)
        use_hints: Whether to request hints on failures (default: True)
    
    Returns:
        Dict with attempt results and metrics
    """
    persona = api_client.persona
    
    # Start session
    session_id = await api_client.start_session(problem)
    if not session_id:
        return {
            "success": False,
            "attempts": 0,
            "hints_used": 0,
            "session_id": None,
            "error": "Failed to start session"
        }
    
    # Track attempt metrics
    attempt_number = 0
    success = False
    hints_received = []
    last_code = None
    last_result = None
    submission_id = None
    previous_errors = None
    previous_hint_text = None
    logger = get_logger()
    
    while attempt_number < max_attempts and not success:
        attempt_number += 1
        persona.total_attempts += 1
        
        # Generate code using cognitive modeling
        code = await generate_code_with_cognitive_modeling(
            persona=persona,
            problem=problem,
            attempt_number=attempt_number,
            previous_errors=previous_errors,
            previous_code=last_code,
            previous_hint=previous_hint_text
        )
        last_code = code
        
        # Submit via API (automatic execution + learner state update)
        result = await api_client.submit_code(
            problem=problem,
            code=code,
            language="python"
        )
        
        last_result = result
        success = result.get("success", False)
        submission_id = result.get("submission_id")
        previous_errors = result.get("error_message")
        
        # Log submission result
        logger.log_submission(
            persona_key=persona.user_key,
            problem_title=problem.get('title', 'Untitled'),
            attempt=attempt_number,
            success=success,
            passed=result.get("passed_count", 0),
            total=result.get("total_count", 0),
            error_message=previous_errors
        )
        
        # Agent interactions based on result
        if not success and use_hints and attempt_number < max_attempts:
            base_hint_prob = 0.80
            attempt_boost = attempt_number * 0.10
            should_ask = random.random() < min(0.95, base_hint_prob + attempt_boost)
            
            if should_ask:
                hint = await api_client.request_hint(
                    problem=problem,
                    code=code,
                    hint_level=None
                )
                
                if hint.get("success"):
                    hints_received.append(hint)
                    persona.hints_requested += 1
                    previous_hint_text = hint.get("hint_text", hint.get("hint", ""))
                    
                    logger.log_agent_interaction(
                        agent_type="hint",
                        persona_key=persona.user_key,
                        action=f"Level {hint.get('hint_level', '?')}",
                        success=True,
                        metadata={"attempt": attempt_number}
                    )
                
                if attempt_number >= 2 and random.random() < 0.5:
                    question = f"I'm struggling with {problem.get('title', 'this problem')}. {previous_errors[:100] if previous_errors else 'My code is failing'}"
                    orch_result = await api_client.ask_orchestrator(
                        question=question,
                        context={"problem_id": problem['question_id'], "attempt": attempt_number}
                    )
                    
                    if orch_result.get("success"):
                        logger.log_agent_interaction(
                            agent_type="orchestrator",
                            persona_key=persona.user_key,
                            action="Conceptual help",
                            success=True
                        )
        
        elif success:
            if submission_id and random.random() < 0.7:
                analysis_result = await api_client.request_code_analysis(
                    problem=problem,
                    code=code,
                    submission_id=submission_id
                )
                
                if analysis_result.get("success"):
                    logger.log_agent_interaction(
                        agent_type="analysis",
                        persona_key=persona.user_key,
                        action="Code review",
                        success=True
                    )
        
        await asyncio.sleep(0.1)
    
    # Update success counter
    if success:
        persona.successful_attempts += 1
    
    # End session
    await api_client.end_session()
    
    return {
        "success": success,
        "attempts": attempt_number,
        "hints_used": len(hints_received),
        "hints_received": hints_received,
        "session_id": session_id,
        "submission_id": submission_id,
        "final_result": last_result
    }


async def simulate_14_day_trajectory(
    persona: SyntheticPersona,
    questions: List[Dict[str, Any]],
    save_trajectory: bool = True,
    output_dir: str = "results/trajectories"
) -> List[DailyMetrics]:
    """
    Simulate 14-day learning trajectory for a single persona using backend API.
    
    This is the core simulation loop:
    - Login once at start
    - For each day:
      - Select problem based on mastery
      - Create session
      - Make multiple attempts with hints
      - Submit code (automatic learner state updates)
      - End session
    - All metrics tracked automatically by backend!
    
    Args:
        persona: SyntheticPersona (must have database_user_key set)
        questions: List of available questions
        save_trajectory: If True, save daily metrics to file (default: True)
        output_dir: Directory to save trajectory files
    
    Returns:
        List of DailyMetrics for each day
    """
    print(f"\n{'='*60}")
    print(f"Starting 14-Day Simulation: {persona.user_key}")
    print(f"Skill Level: {persona.skill_level}")
    print(f"Initial Avg Mastery: {get_persona_average_mastery(persona):.3f}")
    print(f"{'='*60}\n")
    
    # Create API client and login
    api_client = IntelliTAPIClient(persona)
    login_success = await api_client.login()
    
    if not login_success:
        print(f"❌ Failed to login for {persona.user_key}, aborting simulation")
        return []
    
    trajectory = []
    attempted_questions = set()
    logger = get_logger()
    
    for day in range(1, 15):
        metrics = DailyMetrics(day, persona.user_key)
        
        # Consistency check - does persona practice today?
        if random.random() > persona.consistency:
            metrics.active = False
            metrics.skipped_reason = "consistency_check_failed"
            print(f"Day {day}: {persona.user_key} skipped (consistency)")
            trajectory.append(metrics)
            continue
        
        metrics.active = True
        persona.days_active += 1
        
        # Capture mastery before attempt
        metrics.mastery_before = persona.current_mastery.copy()
        
        # Select problem
        problem = select_problem_for_persona(
            persona=persona,
            questions=questions,
            day=day,
            attempted_questions=attempted_questions
        )
        
        if not problem:
            metrics.skipped_reason = "no_suitable_problem"
            print(f"Day {day}: {persona.user_key} - No suitable problem found")
            trajectory.append(metrics)
            continue
        
        # Track problem info
        problem_id = problem.get('question_id')
        metrics.problem_attempted = problem.get('title', 'Untitled')
        metrics.problem_id = problem_id
        metrics.problem_difficulty = problem.get('difficulty', 'Medium')
        metrics.problem_topics = problem.get('topics', [])
        attempted_questions.add(problem_id)
        
        print(f"\nDay {day}: {metrics.problem_attempted} ({metrics.problem_difficulty})")
        
        # Attempt problem (uses API, automatic learner state updates!)
        try:
            attempt_result = await simulate_problem_attempt(
                api_client=api_client,
                problem=problem,
                max_attempts=3,
                use_hints=True
            )
            
            # Extract metrics
            metrics.attempts = attempt_result.get("attempts", 0)
            metrics.hints_used = attempt_result.get("hints_used", 0)
            metrics.hints_received = attempt_result.get("hints_received", [])
            metrics.success = attempt_result.get("success", False)
            metrics.session_id = attempt_result.get("session_id")
            metrics.submission_id = attempt_result.get("submission_id")
            
        except Exception as e:
            print(f"  ❌ Exception during problem attempt: {e}")
            import traceback
            traceback.print_exc()
            metrics.skipped_reason = f"exception: {str(e)}"
        
        # Update persona mastery using cognitive state update
        # This tracks learning progression over time
        update_cognitive_state(
            persona=persona,
            problem=problem,
            success=metrics.success,
            attempts=metrics.attempts,
            hints_used=metrics.hints_used,
            time_spent_seconds=0  # Could track actual time if needed
        )
        
        # Also update using old mastery function for backward compatibility
        # (cognitive state update is more sophisticated)
        topics = problem.get('topics', [])
        if topics:
            difficulty_weight = {
                "Easy": 0.8,
                "Medium": 1.0,
                "Hard": 1.2
            }.get(metrics.problem_difficulty, 1.0)
            
            # Note: update_cognitive_state already updated mastery,
            # but update_persona_mastery provides additional tracking
            # for problem selection heuristics
        
        # Capture mastery after attempt
        metrics.mastery_after = persona.current_mastery.copy()
        
        # Calculate mastery deltas
        for topic in metrics.mastery_after:
            before = metrics.mastery_before.get(topic, 0)
            after = metrics.mastery_after[topic]
            metrics.mastery_delta[topic] = round(after - before, 4)
        
        trajectory.append(metrics)
        
        # Log day summary
        mastery_change = get_persona_average_mastery(persona) - (
            sum(metrics.mastery_before.values()) / len(metrics.mastery_before) 
            if metrics.mastery_before else 0
        )
        
        logger.log_day_summary(
            day=day,
            persona_key=persona.user_key,
            problem_title=metrics.problem_attempted,
            attempts=metrics.attempts,
            success=metrics.success,
            hints_used=metrics.hints_used,
            mastery_change=mastery_change
        )
        
        await asyncio.sleep(0.1)
    
    # Save trajectory to file
    if save_trajectory:
        try:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{persona.user_key}_trajectory.json")
            
            initial_avg_mastery = (
                sum(persona.initial_mastery.values()) / len(persona.initial_mastery)
                if persona.initial_mastery else 0
            )
            final_avg_mastery = get_persona_average_mastery(persona)
            
            trajectory_data = {
                "persona_key": persona.user_key,
                "skill_level": persona.skill_level,
                "simulation_start": trajectory[0].timestamp.isoformat() if trajectory else None,
                "simulation_end": trajectory[-1].timestamp.isoformat() if trajectory else None,
                "total_days": len([m for m in trajectory if m.active]),
                "total_attempts": sum(m.attempts for m in trajectory),
                "total_successes": sum(1 for m in trajectory if m.success),
                "total_hints_used": sum(m.hints_used for m in trajectory),
                "initial_avg_mastery": initial_avg_mastery,
                "final_avg_mastery": final_avg_mastery,
                "mastery_gain": final_avg_mastery - initial_avg_mastery,
                "daily_metrics": [m.to_dict() for m in trajectory]
            }
            
            with open(filepath, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            
            print(f"\n✅ Saved trajectory to {filepath}")
            
        except Exception as e:
            print(f"\n⚠️ Failed to save trajectory: {e}")
    
    initial_avg = (
        sum(persona.initial_mastery.values()) / len(persona.initial_mastery)
        if persona.initial_mastery else 0
    )
    final_avg = get_persona_average_mastery(persona)
    success_rate = persona.successful_attempts / persona.total_attempts if persona.total_attempts > 0 else 0
    
    logger.log_simulation_complete(
        persona_key=persona.user_key,
        days_active=persona.days_active,
        total_attempts=persona.total_attempts,
        success_rate=success_rate,
        mastery_gain=final_avg - initial_avg,
        total_hints=sum(m.hints_used for m in trajectory)
    )
    
    return trajectory


async def run_parallel_simulations(
    personas: List[SyntheticPersona],
    questions: List[Dict[str, Any]],
    max_concurrent: int = 5
) -> Dict[str, List[DailyMetrics]]:
    """
    Run simulations for multiple personas in parallel.
    
    Each persona runs independently with their own API client and login.
    
    RATE LIMITING:
    - All concurrent tasks share a GLOBAL rate limiter (10 RPM hard limit)
    - With max_concurrent=2: each task gets ~5 requests/min on average
    - Rate limiter automatically blocks tasks when limit reached
    - This prevents 429 errors from Gemini API
    
    Args:
        personas: List of SyntheticPersona objects (with database_user_key set)
        questions: List of questions
        max_concurrent: Maximum concurrent simulations (default: 5)
    
    Returns:
        Dict mapping persona_key to trajectory (list of DailyMetrics)
    """
    print(f"\n{'='*80}")
    print(f"Running Parallel Simulations for {len(personas)} Personas")
    print(f"Max Concurrent: {max_concurrent}")
    print(f"Questions Available: {len(questions)}")
    print(f"{'='*80}\n")
    
    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(persona):
        async with semaphore:
            trajectory = await simulate_14_day_trajectory(
                persona=persona,
                questions=questions,
                save_trajectory=True
            )
            return persona.user_key, trajectory
    
    # Create tasks
    tasks = [run_with_semaphore(p) for p in personas]
    
    # Run with progress tracking
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        persona_key, trajectory = await task
        results[persona_key] = trajectory
        print(f"\n[{i}/{len(personas)}] Completed simulation for {persona_key}")
    
    print(f"\n{'='*80}")
    print(f"✅ All {len(personas)} Simulations Complete")
    print(f"{'='*80}\n")
    
    return results
