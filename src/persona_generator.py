"""
Synthetic Persona Generator - Create diverse learner profiles for evaluation.

Generates 20-30 synthetic users with realistic cognitive and behavioral parameters
to evaluate IntelliT's multi-agent teaching system.

Distribution:
- 40% beginners (mastery < 0.3)
- 40% intermediate (0.3-0.7)
- 20% advanced (0.7+)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime


# Topic list from IntelliT system (DSA topics)
DEFAULT_TOPICS = [
    "arrays",
    "strings", 
    "linked-lists",
    "stacks",
    "queues",
    "trees",
    "graphs",
    "hashing",
    "heaps",
    "dynamic-programming",
    "greedy",
    "backtracking",
    "recursion",
    "sorting",
    "searching",
    "two-pointers",
    "sliding-window",
    "binary-search"
]


@dataclass
class SyntheticPersona:
    """
    Synthetic learner profile with cognitive and behavioral parameters.
    
    Attributes:
        user_key: Unique identifier (e.g., "synthetic_beginner_001")
        skill_level: "beginner", "intermediate", or "advanced"
        
        # Cognitive parameters (for student_sim)
        initial_mastery: Topic -> mastery level (0-1)
        learning_rate: How fast they improve (0.05-0.20)
        mistake_patterns: Common error types they make
        working_memory: Affects complex problem performance (0.5-1.0)
        metacognition: Self-awareness for hints (0.3-1.0)
        
        # Behavioral parameters
        consistency: Daily practice probability (0.6-0.95)
        hint_reliance: Probability of requesting hints (0.2-0.8)
        session_duration_minutes: Typical session length (30-120)
        fatigue_rate: Performance decay over session (0.0-0.3)
        
        # IntelliT integration
        database_user_key: Actual ArangoDB user._key (set after DB creation)
        current_mastery: Tracked mastery (updated during simulation)
    """
    
    user_key: str
    skill_level: str
    
    # Cognitive parameters
    initial_mastery: Dict[str, float]
    learning_rate: float
    mistake_patterns: List[str]
    working_memory: float
    metacognition: float
    
    # Behavioral parameters
    consistency: float
    hint_reliance: float
    session_duration_minutes: int
    fatigue_rate: float
    
    # Integration fields
    database_user_key: Optional[str] = None
    current_mastery: Dict[str, float] = field(default_factory=dict)
    
    # Tracking fields
    total_attempts: int = 0
    successful_attempts: int = 0
    hints_requested: int = 0
    days_active: int = 0
    
    def __post_init__(self):
        """Initialize current_mastery from initial_mastery."""
        if not self.current_mastery:
            self.current_mastery = self.initial_mastery.copy()


def generate_diverse_personas(
    n: int = 25,
    topics: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> List[SyntheticPersona]:
    """
    Generate diverse synthetic learner personas across proficiency bands.
    
    Distribution:
    - 40% beginners (skill_level < 0.3)
    - 40% intermediate (0.3 <= skill_level < 0.7)
    - 20% advanced (skill_level >= 0.7)
    
    Args:
        n: Number of personas to generate (default: 25)
        topics: List of topic names (default: DEFAULT_TOPICS)
        seed: Random seed for reproducibility (default: None)
    
    Returns:
        List of SyntheticPersona objects
    """
    if seed is not None:
        np.random.seed(seed)
    
    if topics is None:
        topics = DEFAULT_TOPICS
    
    personas = []
    
    # Calculate counts for each skill level
    n_beginners = int(n * 0.4)
    n_intermediate = int(n * 0.4)
    n_advanced = n - n_beginners - n_intermediate  # Remaining
    
    # Generate beginners
    for i in range(n_beginners):
        persona = _generate_beginner_persona(i + 1, topics)
        personas.append(persona)
    
    # Generate intermediate learners
    for i in range(n_intermediate):
        persona = _generate_intermediate_persona(i + 1, topics)
        personas.append(persona)
    
    # Generate advanced learners
    for i in range(n_advanced):
        persona = _generate_advanced_persona(i + 1, topics)
        personas.append(persona)
    
    print(f"✅ Generated {len(personas)} synthetic personas:")
    print(f"   - {n_beginners} beginners")
    print(f"   - {n_intermediate} intermediate")
    print(f"   - {n_advanced} advanced")
    
    return personas


def _generate_beginner_persona(idx: int, topics: List[str]) -> SyntheticPersona:
    """Generate a beginner-level persona."""
    base_mastery = np.random.uniform(0.05, 0.25)
    
    # Beginners have varied topic strengths with low overall mastery
    initial_mastery = {}
    for topic in topics:
        # Add random variation ±0.15 around base
        mastery = np.clip(base_mastery + np.random.uniform(-0.15, 0.15), 0.0, 0.4)
        initial_mastery[topic] = round(mastery, 3)
    
    # Common beginner mistakes
    mistake_patterns = [
        "syntax-error",
        "off-by-one",
        "null-check",
        "logic-error",
        "incomplete-solution",
        "variable-scope",
        "type-mismatch"
    ]
    
    return SyntheticPersona(
        user_key=f"synthetic_beginner_{idx:03d}",
        skill_level="beginner",
        initial_mastery=initial_mastery,
        learning_rate=np.random.uniform(0.08, 0.12),  # Learn relatively fast (starting from low)
        mistake_patterns=np.random.choice(mistake_patterns, size=5, replace=False).tolist(),
        working_memory=np.random.uniform(0.5, 0.7),
        metacognition=np.random.uniform(0.3, 0.5),
        consistency=np.random.uniform(0.6, 0.8),
        hint_reliance=np.random.uniform(0.65, 0.85),  # High reliance on hints
        session_duration_minutes=int(np.random.uniform(30, 60)),
        fatigue_rate=np.random.uniform(0.15, 0.25)  # Fatigue faster
    )


def _generate_intermediate_persona(idx: int, topics: List[str]) -> SyntheticPersona:
    """Generate an intermediate-level persona."""
    base_mastery = np.random.uniform(0.4, 0.6)
    
    # Intermediate learners have some strong topics, some weak
    initial_mastery = {}
    for topic in topics:
        mastery = np.clip(base_mastery + np.random.uniform(-0.2, 0.2), 0.2, 0.8)
        initial_mastery[topic] = round(mastery, 3)
    
    # Intermediate mistakes - more algorithmic
    mistake_patterns = [
        "edge-case",
        "optimization",
        "off-by-one",
        "algorithm-choice",
        "complexity-issue"
    ]
    
    return SyntheticPersona(
        user_key=f"synthetic_intermediate_{idx:03d}",
        skill_level="intermediate",
        initial_mastery=initial_mastery,
        learning_rate=np.random.uniform(0.12, 0.18),  # Moderate learning rate
        mistake_patterns=np.random.choice(mistake_patterns, size=3, replace=False).tolist(),
        working_memory=np.random.uniform(0.65, 0.85),
        metacognition=np.random.uniform(0.5, 0.75),
        consistency=np.random.uniform(0.75, 0.90),
        hint_reliance=np.random.uniform(0.35, 0.55),  # Moderate hint usage
        session_duration_minutes=int(np.random.uniform(45, 90)),
        fatigue_rate=np.random.uniform(0.08, 0.15)
    )


def _generate_advanced_persona(idx: int, topics: List[str]) -> SyntheticPersona:
    """Generate an advanced-level persona."""
    base_mastery = np.random.uniform(0.70, 0.85)
    
    # Advanced learners have high mastery across most topics
    initial_mastery = {}
    for topic in topics:
        mastery = np.clip(base_mastery + np.random.uniform(-0.15, 0.10), 0.5, 0.95)
        initial_mastery[topic] = round(mastery, 3)
    
    # Advanced mistakes - subtle issues
    mistake_patterns = [
        "optimization",
        "edge-case",
        "corner-case",
        "numerical-precision"
    ]
    
    return SyntheticPersona(
        user_key=f"synthetic_advanced_{idx:03d}",
        skill_level="advanced",
        initial_mastery=initial_mastery,
        learning_rate=np.random.uniform(0.05, 0.10),  # Slower (diminishing returns)
        mistake_patterns=np.random.choice(mistake_patterns, size=2, replace=False).tolist(),
        working_memory=np.random.uniform(0.80, 1.0),
        metacognition=np.random.uniform(0.75, 1.0),
        consistency=np.random.uniform(0.85, 0.95),
        hint_reliance=np.random.uniform(0.15, 0.30),  # Low hint usage
        session_duration_minutes=int(np.random.uniform(60, 120)),
        fatigue_rate=np.random.uniform(0.03, 0.10)  # Resistant to fatigue
    )


def get_persona_average_mastery(persona: SyntheticPersona) -> float:
    """Calculate average mastery across all topics."""
    if not persona.current_mastery:
        return 0.0
    return np.mean(list(persona.current_mastery.values()))


def get_persona_topic_strengths(persona: SyntheticPersona, top_n: int = 5) -> List[tuple]:
    """
    Get persona's strongest topics.
    
    Args:
        persona: SyntheticPersona
        top_n: Number of top topics to return
    
    Returns:
        List of (topic, mastery) tuples sorted by mastery descending
    """
    if not persona.current_mastery:
        return []
    
    sorted_topics = sorted(
        persona.current_mastery.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_topics[:top_n]


def get_persona_topic_weaknesses(persona: SyntheticPersona, bottom_n: int = 5) -> List[tuple]:
    """
    Get persona's weakest topics.
    
    Args:
        persona: SyntheticPersona
        bottom_n: Number of bottom topics to return
    
    Returns:
        List of (topic, mastery) tuples sorted by mastery ascending
    """
    if not persona.current_mastery:
        return []
    
    sorted_topics = sorted(
        persona.current_mastery.items(),
        key=lambda x: x[1]
    )
    return sorted_topics[:bottom_n]


def update_persona_mastery(
    persona: SyntheticPersona,
    topic: str,
    success: bool,
    difficulty_weight: float = 1.0
) -> float:
    """
    Update persona's mastery for a topic based on attempt result.
    
    Uses simplified BKT-style update:
    - Success: mastery += learning_rate * difficulty_weight * (1 - mastery)
    - Failure: mastery -= 0.15 * difficulty_weight * mastery
    
    Args:
        persona: SyntheticPersona to update
        topic: Topic that was practiced
        success: Whether attempt was successful
        difficulty_weight: Weight based on problem difficulty (0.8-1.2)
    
    Returns:
        Updated mastery value
    """
    if topic not in persona.current_mastery:
        persona.current_mastery[topic] = 0.1
    
    current = persona.current_mastery[topic]
    
    if success:
        # Increase mastery with diminishing returns
        delta = persona.learning_rate * difficulty_weight * (1.0 - current)
        new_mastery = min(1.0, current + delta)
    else:
        # Decrease mastery proportionally
        delta = 0.15 * difficulty_weight * current
        new_mastery = max(0.0, current - delta)
    
    persona.current_mastery[topic] = round(new_mastery, 3)
    return new_mastery


def should_request_hint(persona: SyntheticPersona, current_attempt: int) -> bool:
    """
    Determine if persona should request a hint based on their profile.
    
    Probability increases with:
    - Higher hint_reliance
    - More failed attempts
    - Lower metacognition
    
    Args:
        persona: SyntheticPersona
        current_attempt: Number of attempts so far on this problem
    
    Returns:
        True if should request hint
    """
    # Base probability from persona's hint_reliance
    base_prob = persona.hint_reliance
    
    # Increase probability with more attempts
    attempt_modifier = min(0.2, current_attempt * 0.05)
    
    # Lower metacognition = more likely to ask for help
    metacog_modifier = (1.0 - persona.metacognition) * 0.1
    
    total_prob = min(0.95, base_prob + attempt_modifier + metacog_modifier)
    
    return np.random.random() < total_prob


def determine_hint_level(persona: SyntheticPersona, attempt_number: int) -> int:
    """
    Determine appropriate hint level (1-5) based on persona and attempts.
    
    Args:
        persona: SyntheticPersona
        attempt_number: Which attempt this is (1, 2, 3, ...)
    
    Returns:
        Hint level (1-5)
    """
    # Start with lower hints, escalate with attempts
    if persona.skill_level == "beginner":
        base_level = min(5, 2 + attempt_number // 2)
    elif persona.skill_level == "intermediate":
        base_level = min(5, 1 + attempt_number // 2)
    else:  # advanced
        base_level = min(4, 1 + attempt_number // 3)
    
    return base_level


def print_persona_summary(persona: SyntheticPersona):
    """Print human-readable summary of a persona."""
    avg_mastery = get_persona_average_mastery(persona)
    
    print(f"\n{'='*60}")
    print(f"Persona: {persona.user_key}")
    print(f"{'='*60}")
    print(f"Skill Level: {persona.skill_level.title()}")
    print(f"Average Mastery: {avg_mastery:.3f}")
    print(f"Learning Rate: {persona.learning_rate:.3f}")
    print(f"Consistency: {persona.consistency:.2f}")
    print(f"Hint Reliance: {persona.hint_reliance:.2f}")
    print(f"Working Memory: {persona.working_memory:.2f}")
    print(f"Metacognition: {persona.metacognition:.2f}")
    print(f"\nMistake Patterns: {', '.join(persona.mistake_patterns)}")
    
    if persona.current_mastery:
        strengths = get_persona_topic_strengths(persona, top_n=3)
        weaknesses = get_persona_topic_weaknesses(persona, bottom_n=3)
        
        print(f"\nTop 3 Strengths:")
        for topic, mastery in strengths:
            print(f"  - {topic}: {mastery:.3f}")
        
        print(f"\nTop 3 Weaknesses:")
        for topic, mastery in weaknesses:
            print(f"  - {topic}: {mastery:.3f}")
    
    print(f"{'='*60}\n")

