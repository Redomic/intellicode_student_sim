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


async def generate_code_with_llm(
    persona: SyntheticPersona,
    problem: Dict[str, Any],
    attempt_number: int,
    previous_errors: Optional[str] = None,
    previous_code: Optional[str] = None
) -> str:
    """
    Generate REAL code attempt using LLM based on persona's cognitive state.
    
    Uses Gemini to simulate realistic student attempts with appropriate mistakes.
    NOT hardcoded templates - actual problem-solving attempts.
    
    Args:
        persona: SyntheticPersona with cognitive profile
        problem: Question dict with problem statement
        attempt_number: Current attempt number (1-indexed)
        previous_errors: Error message from previous attempt
        previous_code: Previous code attempt
    
    Returns:
        Python code string generated by LLM
    """
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Get Gemini API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    
    # Initialize Gemini (fast, cheap for synthetic users)
    llm = ChatGoogleGenerativeAI(
        model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'),
        google_api_key=api_key,
        temperature=0.7,
        max_output_tokens=8192  # Increased to avoid truncation
    )
    
    # Extract rich problem data from roadmap structure
    problem_text = problem.get('question', problem.get('problem_statement_text', ''))
    examples = problem.get('original_data', {}).get('examples', [])
    constraints = problem.get('original_data', {}).get('constraints', [])
    test_cases = problem.get('sample_test_cases', [])
    
    # Build structured problem description
    problem_description = f"# Problem: {problem.get('title', 'Untitled')}\n\n{problem_text}\n"
    
    # Add examples if available
    if examples:
        problem_description += "\n## Examples:\n"
        for i, ex in enumerate(examples[:3], 1):  # Show up to 3 examples
            if isinstance(ex, dict):
                problem_description += f"Example {i}:\n"
                if 'input' in ex:
                    problem_description += f"Input: {ex['input']}\n"
                if 'output' in ex:
                    problem_description += f"Output: {ex['output']}\n"
                if 'explanation' in ex:
                    problem_description += f"Explanation: {ex['explanation']}\n"
                problem_description += "\n"
    
    # Add test cases if examples not available
    elif test_cases:
        problem_description += "\n## Test Cases:\n"
        for i, tc in enumerate(test_cases[:2], 1):  # Show 2 test cases
            problem_description += f"Test {i}:\n"
            problem_description += f"Input: {tc.get('input', 'N/A')}\n"
            problem_description += f"Expected: {tc.get('expected_output', 'N/A')}\n\n"
    
    # Add constraints if available
    if constraints:
        problem_description += "\n## Constraints:\n"
        for constraint in constraints[:5]:  # Show up to 5 constraints
            problem_description += f"- {constraint}\n"
    
    # Skill-level specific persona instructions
    if persona.skill_level == "beginner":
        skill_instruction = """You are simulating a BEGINNER programmer learning DSA.

Common mistakes you make:
- Forget to handle edge cases (empty arrays, single elements)
- Make syntax errors (missing colons, wrong indentation)
- Confuse variable names and scopes
- Use wrong operators or logic
- Don't validate inputs
- Inefficient brute force approaches

Write code that ATTEMPTS the solution but has 1-2 realistic beginner mistakes."""

    elif persona.skill_level == "intermediate":
        skill_instruction = """You are simulating an INTERMEDIATE programmer with DSA knowledge.

Common mistakes you make:
- Off-by-one errors in loops
- Miss edge cases (but handle basic ones)
- Choose suboptimal algorithms
- Logic errors in conditionals
- Incorrect handling of data structure boundaries

Write code with good structure but include 1 subtle intermediate-level bug."""

    else:  # advanced
        skill_instruction = """You are simulating an ADVANCED programmer skilled in DSA.

Common mistakes you make:
- Very rare edge cases (like integer overflow edge cases)
- Subtle algorithmic optimizations missed
- Minor logic errors in complex conditions

Write mostly correct, efficient code with at most 1 very subtle issue or be completely correct."""
    
    # Build the complete prompt
    if attempt_number == 1:
        # First attempt - fresh solution
        prompt = f"""{skill_instruction}

{problem_description}

CRITICAL CODE FORMAT:
Your code MUST follow this exact structure:

class Solution:
    def solution(self, param1, param2, ...):
        # Your implementation here
        # IMPORTANT: Always return the result
        # Even for in-place modifications, return the modified array/structure
        return result

IMPORTANT INSTRUCTIONS:
1. Code MUST be inside a class called 'Solution'
2. The method MUST be named 'solution' and take 'self' as first parameter
3. Infer the correct parameter names and types from the examples/test cases
4. ALWAYS return the result - even for in-place modifications, return the modified array
5. Write ONLY executable Python code in the Solution class format
6. Do NOT include explanations, brief comments are ok
7. Make realistic mistakes based on your skill level

Generate the Solution class code now:"""

    else:
        # Retry attempt - fix previous errors
        prompt = f"""{skill_instruction}

{problem_description}

## Your Previous Attempt:
```python
{previous_code}
```

## Error/Failure:
{previous_errors}

CRITICAL CODE FORMAT:
Your code MUST follow this structure:

class Solution:
    def solution(self, param1, param2, ...):
        # Fixed implementation
        # IMPORTANT: Always return the result
        return result

INSTRUCTIONS:
1. Fix the error in your previous code
2. Keep the Solution class structure with solution method
3. The method must have 'self' as first parameter
4. ALWAYS return the result - even for in-place modifications, return the modified array
5. Write ONLY the corrected executable Python code
6. No explanations needed

Generate the corrected Solution class code now:"""
    
    # Log the LLM call for transparency
    print(f"\n{'='*60}")
    print(f"📤 LLM CODE GENERATION REQUEST")
    print(f"{'='*60}")
    print(f"Model: {llm.model}")
    print(f"Persona: {persona.user_key} ({persona.skill_level})")
    print(f"Attempt: {attempt_number}")
    print(f"Problem: {problem.get('title', 'Untitled')}")
    print(f"\nPrompt length: {len(prompt)} chars")
    print(f"{'='*60}\n")
    
    # Generate code with exponential backoff for empty responses
    max_retries = 5  # Increased from 3
    base_delay = 2  # Start with 2 seconds
    
    for retry in range(max_retries):
        try:
            messages = [("human", prompt)]
            response = await llm.ainvoke(messages)
            
            code = response.content.strip() if response.content else ""
            
            # Log the response
            print(f"\n{'='*60}")
            print(f"📥 LLM CODE GENERATION RESPONSE (Retry {retry + 1}/{max_retries})")
            print(f"{'='*60}")
            print(f"Raw response length: {len(code)} chars")
            if code:
                print(f"\nGenerated code preview:")
                print(code[:300] + "..." if len(code) > 300 else code)
            else:
                print("\n⚠️ EMPTY RESPONSE FROM LLM - Possible safety filter or rate limit")
            print(f"{'='*60}\n")
            
            # Check if response is empty
            if not code or len(code) < 10:
                # Exponential backoff: 2s, 4s, 8s, 16s, 32s
                delay = base_delay * (2 ** retry)
                print(f"  ⚠️ Empty or too short response, waiting {delay}s before retry ({retry + 1}/{max_retries})...")
                await asyncio.sleep(delay)
                continue
            
            # Extract code if wrapped in markdown
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            # Validate we got actual code with class definition
            if not code or len(code) < 10:
                delay = base_delay * (2 ** retry)
                print(f"  ⚠️ Code extraction failed, waiting {delay}s before retry ({retry + 1}/{max_retries})...")
                await asyncio.sleep(delay)
                continue
            
            # Verify it has the Solution class
            if "class Solution:" not in code:
                print(f"  ⚠️ Generated code missing 'class Solution:', attempting extraction...")
                # Try to fix common issues
                if "class" in code.lower():
                    print(f"  ℹ️ Found class but wrong name, will attempt anyway")
            
            return code
            
        except Exception as e:
            print(f"  ❌ Error generating code: {e}")
            if retry < max_retries - 1:
                delay = base_delay * (2 ** retry)
                print(f"  🔄 Retrying in {delay}s ({retry + 2}/{max_retries})...")
                await asyncio.sleep(delay)
            else:
                raise
    
    # If all retries failed, return minimal fallback
    raise ValueError(f"Failed to generate code after {max_retries} retries - all responses were empty or invalid")


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
    
    print(f"\n  🧠 Generating code with LLM (skill: {persona.skill_level})...")
    
    while attempt_number < max_attempts and not success:
        attempt_number += 1
        persona.total_attempts += 1
        
        # Generate REAL code using LLM based on cognitive state
        print(f"  🤖 Attempt {attempt_number}: Calling Gemini to generate code...")
        code = await generate_code_with_llm(
            persona=persona,
            problem=problem,
            attempt_number=attempt_number,
            previous_errors=previous_errors,
            previous_code=last_code
        )
        print(f"  ✅ Generated {len(code)} chars of code")
        print(f"  📝 Code preview: {code[:100]}...")
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
        
        # If failed and hints available, request hint
        if not success and use_hints and attempt_number < max_attempts:
            if should_request_hint(persona, attempt_number):
                hint = await api_client.request_hint(
                    problem=problem,
                    code=code,
                    hint_level=None  # Let backend auto-calculate
                )
                
                if hint.get("success"):
                    hints_received.append(hint)
                    persona.hints_requested += 1
        
        # Small delay between attempts
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
    output_dir: str = "student_sim/results/trajectories"
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
    attempted_questions = set()  # Track to avoid immediate repeats
    
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
        
        print(f"Day {day}: {persona.user_key} attempting '{metrics.problem_attempted}' ({metrics.problem_difficulty})")
        
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
        
        # Update persona mastery (local tracking for problem selection)
        topics = problem.get('topics', [])
        if topics:
            difficulty_weight = {
                "Easy": 0.8,
                "Medium": 1.0,
                "Hard": 1.2
            }.get(metrics.problem_difficulty, 1.0)
            
            for topic in topics:
                update_persona_mastery(
                    persona=persona,
                    topic=topic,
                    success=metrics.success,
                    difficulty_weight=difficulty_weight
                )
        
        # Capture mastery after attempt
        metrics.mastery_after = persona.current_mastery.copy()
        
        # Calculate mastery deltas
        for topic in metrics.mastery_after:
            before = metrics.mastery_before.get(topic, 0)
            after = metrics.mastery_after[topic]
            metrics.mastery_delta[topic] = round(after - before, 4)
        
        trajectory.append(metrics)
        
        print(f"  Final: {'✅ SUCCESS' if metrics.success else '❌ FAILED'}, "
              f"Avg Mastery: {get_persona_average_mastery(persona):.3f}, "
              f"Hints: {metrics.hints_used}")
        
        # Small delay between days
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
    
    print(f"\n{'='*60}")
    print(f"Completed 14-Day Simulation: {persona.user_key}")
    print(f"Days Active: {persona.days_active}/14")
    print(f"Success Rate: {persona.successful_attempts}/{persona.total_attempts}")
    
    initial_avg = (
        sum(persona.initial_mastery.values()) / len(persona.initial_mastery)
        if persona.initial_mastery else 0
    )
    print(f"Mastery Gain: {get_persona_average_mastery(persona) - initial_avg:.3f}")
    print(f"{'='*60}\n")
    
    return trajectory


async def run_parallel_simulations(
    personas: List[SyntheticPersona],
    questions: List[Dict[str, Any]],
    max_concurrent: int = 5
) -> Dict[str, List[DailyMetrics]]:
    """
    Run simulations for multiple personas in parallel.
    
    Each persona runs independently with their own API client and login.
    
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
