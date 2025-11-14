"""
Cognitive Behavior Prediction Module

Predicts student errors based on cognitive attributes, topic mastery,
and mistake patterns - inspired by the student_sim research paper but
adapted for IntelliT's Python DSA problems without requiring historical data.

Core Methodology:
- Uses persona's cognitive profile (mastery, working memory, metacognition)
- Predicts error probability based on topic weakness
- Suggests specific mistake types to inject
- Tracks learning progression over time
"""

from typing import Dict, Any, List, Optional, Tuple
import random
import numpy as np
from .persona_generator import SyntheticPersona


def predict_student_errors(
    persona: SyntheticPersona,
    problem: Dict[str, Any],
    attempt_number: int,
    previous_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict if student will make errors based on cognitive profile.
    
    This is the core of student_sim's behavior prediction, adapted to work
    without requiring historical training data by using persona attributes directly.
    
    Args:
        persona: SyntheticPersona with cognitive profile
        problem: Problem dictionary with topics
        attempt_number: Current attempt (1-indexed)
        previous_hint: Hint text from previous attempt
    
    Returns:
        Dictionary with error prediction:
        {
            "will_make_error": bool,
            "error_probability": float (0-1),
            "error_types": List[str],
            "affected_topics": List[str],
            "suggested_mistakes": List[str],
            "difficulty_factor": float
        }
    """
    
    # Extract problem topics
    problem_topics = problem.get('topics', [])
    if not problem_topics:
        # Fallback to difficulty-based estimation
        problem_topics = ['general']
    
    # Normalize topic names to match persona mastery keys
    normalized_topics = [normalize_topic_name(t) for t in problem_topics]
    
    # Calculate topic-specific mastery
    topic_masteries = []
    for topic in normalized_topics:
        mastery = persona.current_mastery.get(topic, persona.average_mastery)
        topic_masteries.append(mastery)
    
    avg_topic_mastery = np.mean(topic_masteries) if topic_masteries else persona.average_mastery
    
    # Base error probability from topic weakness
    # Low mastery → high error probability
    base_error_prob = 1.0 - avg_topic_mastery
    
    # Adjust for cognitive attributes
    cognitive_factor = calculate_cognitive_factor(persona)
    error_prob = base_error_prob * (2.0 - cognitive_factor)  # 0.5-1.5x multiplier
    
    # Adjust for attempt number (improve with practice)
    attempt_factor = 1.0 / (1.0 + (attempt_number - 1) * 0.3)
    error_prob *= attempt_factor
    
    # If received hint, reduce error probability
    if previous_hint:
        hint_reduction = persona.hint_benefit * 0.5  # Hints help but not perfectly
        error_prob *= (1.0 - hint_reduction)
    
    # Ensure probability stays in valid range
    error_prob = np.clip(error_prob, 0.05, 0.95)
    
    # Decide if error will occur
    will_make_error = random.random() < error_prob
    
    # Identify weak topics (mastery < 0.4)
    weak_topics = [
        topic for topic, mastery in zip(normalized_topics, topic_masteries)
        if mastery < 0.4
    ]
    
    # Select error types based on mistake patterns and skill level
    error_types = select_error_types(persona, attempt_number, avg_topic_mastery)
    
    # Generate specific mistake suggestions
    suggested_mistakes = generate_mistake_suggestions(
        persona, problem, error_types, weak_topics
    )
    
    # Calculate difficulty factor (problem difficulty vs student ability)
    difficulty_factor = calculate_difficulty_factor(problem, persona)
    
    return {
        "will_make_error": will_make_error,
        "error_probability": float(error_prob),
        "error_types": error_types,
        "affected_topics": weak_topics,
        "suggested_mistakes": suggested_mistakes,
        "difficulty_factor": difficulty_factor,
        "avg_topic_mastery": float(avg_topic_mastery)
    }


def calculate_cognitive_factor(persona: SyntheticPersona) -> float:
    """
    Calculate overall cognitive ability factor.
    
    Higher values → better performance, fewer errors
    Range: 0.3 to 1.0
    """
    factors = [
        persona.working_memory,
        persona.attention_span,
        persona.metacognition,
        persona.consistency
    ]
    
    return np.mean(factors)


def calculate_difficulty_factor(problem: Dict[str, Any], persona: SyntheticPersona) -> float:
    """
    Calculate how difficult this problem is for this specific student.
    
    Returns:
        float: 0.0 (very easy) to 1.0 (very hard)
    """
    difficulty_map = {'Easy': 0.3, 'Medium': 0.6, 'Hard': 0.9}
    base_difficulty = difficulty_map.get(problem.get('difficulty', 'Medium'), 0.6)
    
    # Adjust based on student's topic mastery
    problem_topics = problem.get('topics', [])
    if problem_topics:
        normalized_topics = [normalize_topic_name(t) for t in problem_topics]
        topic_masteries = [
            persona.current_mastery.get(topic, persona.average_mastery)
            for topic in normalized_topics
        ]
        mastery_adjustment = 1.0 - np.mean(topic_masteries)
    else:
        mastery_adjustment = 1.0 - persona.average_mastery
    
    # Combine base difficulty with mastery gap
    difficulty_factor = base_difficulty * 0.5 + mastery_adjustment * 0.5
    
    return np.clip(difficulty_factor, 0.1, 1.0)


def select_error_types(
    persona: SyntheticPersona,
    attempt_number: int,
    topic_mastery: float
) -> List[str]:
    """
    Select which types of errors the student is likely to make.
    
    Based on:
    - Persona's mistake patterns
    - Skill level
    - Attempt number
    - Topic mastery
    """
    
    # Start with persona's characteristic mistake patterns
    available_errors = list(persona.mistake_patterns)
    
    # On first attempt, more errors possible
    # On later attempts, fewer errors (learning effect)
    num_errors = max(1, len(available_errors) // attempt_number)
    
    # Adjust based on topic mastery
    if topic_mastery < 0.3:
        # Very weak → more errors
        num_errors = min(len(available_errors), num_errors + 1)
    elif topic_mastery > 0.7:
        # Strong → fewer errors
        num_errors = max(1, num_errors - 1)
    
    # Select errors randomly from patterns
    selected = random.sample(available_errors, min(num_errors, len(available_errors)))
    
    return selected


def generate_mistake_suggestions(
    persona: SyntheticPersona,
    problem: Dict[str, Any],
    error_types: List[str],
    weak_topics: List[str]
) -> List[str]:
    """
    Generate specific, actionable mistake descriptions for LLM injection.
    
    These are concrete instructions for the error injection LLM.
    """
    
    suggestions = []
    
    for error_type in error_types:
        if error_type == "syntax-error":
            suggestions.extend([
                "Missing colon at end of if/for/while statement",
                "Incorrect indentation in nested blocks",
                "Mismatched parentheses in function call"
            ])
        
        elif error_type == "logic-error":
            suggestions.extend([
                "Wrong comparison operator (< instead of <=)",
                "Incorrect loop termination condition",
                "Flipped if-else logic"
            ])
        
        elif error_type == "edge-case":
            suggestions.extend([
                "Not handling empty array/list",
                "Missing check for single element",
                "No validation for negative numbers"
            ])
        
        elif error_type == "off-by-one":
            suggestions.extend([
                "Loop index off by one (range should be n-1 not n)",
                "Array access out of bounds by one",
                "Incorrect boundary condition in binary search"
            ])
        
        elif error_type == "variable-scope":
            suggestions.extend([
                "Using variable outside its scope",
                "Shadowing outer variable unintentionally",
                "Not initializing variable before use"
            ])
        
        elif error_type == "type-mismatch":
            suggestions.extend([
                "Mixing int and float without conversion",
                "Concatenating string with number",
                "Incorrect type in function parameter"
            ])
        
        elif error_type == "algorithm-choice":
            suggestions.extend([
                "Using O(n²) nested loops instead of O(n) single pass",
                "Not using appropriate data structure (set vs list)",
                "Missing optimization opportunity"
            ])
        
        elif error_type == "incomplete-solution":
            suggestions.extend([
                "Only handling one case, missing other scenarios",
                "Early return without processing all data",
                "Partial implementation of algorithm"
            ])
        
        elif error_type == "recursion-error":
            suggestions.extend([
                "Missing base case in recursion",
                "Incorrect recursive call parameters",
                "Stack overflow due to no termination"
            ])
        
        elif error_type == "optimization":
            suggestions.extend([
                "Inefficient nested loop structure",
                "Redundant calculations in loop",
                "Not using memoization/caching"
            ])
    
    # Add topic-specific mistakes
    for topic in weak_topics:
        if topic == "arrays":
            suggestions.append("Incorrect array indexing or bounds")
        elif topic == "linked-lists":
            suggestions.append("Not handling next pointer correctly")
        elif topic == "trees":
            suggestions.append("Incorrect tree traversal order")
        elif topic == "graphs":
            suggestions.append("Missing visited set in graph traversal")
        elif topic == "dynamic-programming":
            suggestions.append("Not building DP table correctly")
    
    # Randomly select 2-3 specific mistakes to inject
    num_to_select = min(3, len(suggestions))
    if suggestions:
        return random.sample(suggestions, num_to_select)
    
    return ["General logical error in implementation"]


def normalize_topic_name(topic: str) -> str:
    """
    Normalize topic name to match persona mastery keys.
    
    Examples:
        "Two Pointers" → "two-pointers"
        "Dynamic Programming" → "dynamic-programming"
        "Arrays" → "arrays"
    """
    return topic.lower().replace(' ', '-').replace('_', '-')


def calculate_target_success_rate(persona: SyntheticPersona, problem: Dict[str, Any]) -> float:
    """
    Calculate expected success rate for this student on this problem.
    
    Used for calibration and validation.
    
    Returns:
        float: 0.0 to 1.0 expected success probability
    """
    
    # Base success rate by skill level
    skill_base_rates = {
        'beginner': 0.30,      # 30% first-try success
        'intermediate': 0.55,  # 55% first-try success
        'advanced': 0.75       # 75% first-try success
    }
    
    base_rate = skill_base_rates.get(persona.skill_level, 0.50)
    
    # Adjust for cognitive attributes
    cognitive_factor = calculate_cognitive_factor(persona)
    base_rate *= (0.5 + cognitive_factor)  # 0.5x to 1.5x multiplier
    
    # Adjust for topic mastery
    problem_topics = problem.get('topics', [])
    if problem_topics:
        normalized_topics = [normalize_topic_name(t) for t in problem_topics]
        topic_masteries = [
            persona.current_mastery.get(topic, persona.average_mastery)
            for topic in normalized_topics
        ]
        avg_mastery = np.mean(topic_masteries)
    else:
        avg_mastery = persona.average_mastery
    
    # Strong mastery increases success rate
    base_rate *= (0.5 + avg_mastery)  # 0.5x to 1.5x multiplier
    
    # Difficulty adjustment
    difficulty_map = {'Easy': 1.2, 'Medium': 1.0, 'Hard': 0.7}
    difficulty_mult = difficulty_map.get(problem.get('difficulty', 'Medium'), 1.0)
    base_rate *= difficulty_mult
    
    return np.clip(base_rate, 0.05, 0.95)


def update_cognitive_state(
    persona: SyntheticPersona,
    problem: Dict[str, Any],
    success: bool,
    attempts: int,
    hints_used: int,
    time_spent_seconds: int = 0
):
    """
    Update persona's cognitive state based on problem-solving outcome.
    
    This creates learning progression over the 14-day simulation:
    - Success → increase mastery in problem topics
    - Failure → slight decrease (forgetting)
    - Hints → scaled improvement based on hint_benefit
    
    Args:
        persona: SyntheticPersona to update
        problem: Problem that was attempted
        success: Whether problem was solved
        attempts: Number of attempts made
        hints_used: Number of hints requested
        time_spent_seconds: Time spent on problem
    """
    
    # Extract problem topics
    problem_topics = problem.get('topics', [])
    normalized_topics = [normalize_topic_name(t) for t in problem_topics]
    
    # Calculate learning amount based on outcome
    if success:
        # Successful solve → positive learning
        base_gain = persona.learning_rate
        
        # Less gain if used many hints (didn't learn as much independently)
        hint_penalty = 0.2 * hints_used
        learning_gain = base_gain * (1.0 - hint_penalty)
        
        # Bonus for solving on first try
        if attempts == 1:
            learning_gain *= 1.2
        
    else:
        # Failed → small negative adjustment (forgetting or confusion)
        learning_gain = -persona.forgetting_rate * 0.5
    
    # Update mastery for each topic
    for topic in normalized_topics:
        if topic in persona.current_mastery:
            old_mastery = persona.current_mastery[topic]
            new_mastery = old_mastery + learning_gain
            
            # Cap mastery between 0 and 1
            persona.current_mastery[topic] = np.clip(new_mastery, 0.0, 1.0)
    
    # Update statistics
    persona.total_problems_attempted += 1
    if success:
        persona.total_problems_solved += 1
    persona.total_hints_used += hints_used
    
    # Update average mastery
    if persona.current_mastery:
        persona.average_mastery = np.mean(list(persona.current_mastery.values()))


def should_make_mistake_on_retry(
    persona: SyntheticPersona,
    attempt_number: int,
    received_hint: bool
) -> bool:
    """
    Decide if student should still make mistakes on retry attempts.
    
    Even with hints, students don't always fix issues perfectly.
    
    Returns:
        bool: True if student should still make mistakes
    """
    
    # Base probability of continued mistakes
    base_prob = 0.4  # 40% chance of still making errors
    
    # Reduce with each attempt
    prob = base_prob / attempt_number
    
    # If hint received, reduce based on hint_benefit
    if received_hint:
        prob *= (1.0 - persona.hint_benefit * 0.7)
    
    # Metacognition helps avoid repeated mistakes
    prob *= (1.0 - persona.metacognition * 0.3)
    
    return random.random() < prob

