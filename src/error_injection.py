"""
Error Injection Code Generator

Generates code with intentional, realistic mistakes based on cognitive predictions.
Uses LLM with specialized prompts to create student-like code that:
1. Attempts to solve the problem
2. Contains specific predicted errors
3. Shows appropriate skill level
4. Improves with hints (partial incorporation)

This is the core of student_sim's solution simulation, adapted for IntelliT.
"""

from typing import Dict, Any, Optional
import os
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI

from .persona_generator import SyntheticPersona
from .cognitive_behavior import should_make_mistake_on_retry


async def generate_code_with_errors(
    persona: SyntheticPersona,
    problem: Dict[str, Any],
    error_prediction: Dict[str, Any],
    attempt_number: int,
    previous_code: Optional[str] = None,
    previous_errors: Optional[str] = None,
    previous_hint: Optional[str] = None
) -> str:
    """
    Generate code with intentional realistic errors based on cognitive prediction.
    
    This implements the student_sim methodology:
    - First attempt: Inject predicted errors matching persona's weaknesses
    - Later attempts: Show learning progression, incorporate hints partially
    
    Args:
        persona: SyntheticPersona with cognitive profile
        problem: Problem dictionary with full details
        error_prediction: Output from predict_student_errors()
        attempt_number: Current attempt (1-indexed)
        previous_code: Code from previous attempt
        previous_errors: Error message from previous attempt
        previous_hint: Hint text received from IntelliT agents
    
    Returns:
        Python code string with Solution class
    """
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    
    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'),
        google_api_key=api_key,
        temperature=0.8,  # Higher temperature for more variation in mistakes
        max_output_tokens=8192
    )
    
    if attempt_number == 1:
        # First attempt - inject predicted errors
        prompt = build_first_attempt_prompt(persona, problem, error_prediction)
    else:
        # Retry attempt - show learning with hint incorporation
        prompt = build_retry_attempt_prompt(
            persona, problem, error_prediction,
            previous_code, previous_errors, previous_hint,
            attempt_number
        )
    
    # Log prompt for debugging
    print(f"\n{'='*60}")
    print(f"🧠 COGNITIVE CODE GENERATION (Attempt {attempt_number})")
    print(f"{'='*60}")
    print(f"Persona: {persona.user_key} ({persona.skill_level})")
    print(f"Error Probability: {error_prediction['error_probability']:.2f}")
    print(f"Will Make Error: {error_prediction['will_make_error']}")
    print(f"Error Types: {', '.join(error_prediction['error_types'])}")
    print(f"Topic Mastery: {error_prediction['avg_topic_mastery']:.2f}")
    if previous_hint:
        print(f"Hint Received: Yes (Length: {len(previous_hint)} chars)")
    print(f"\nPrompt Length: {len(prompt)} chars")
    print(f"{'='*60}\n")
    
    # Generate code with retries and exponential backoff
    max_retries = 5
    for retry in range(max_retries):
        try:
            messages = [("human", prompt)]
            response = await llm.ainvoke(messages)
            
            code = response.content.strip() if response.content else ""
            
            # Log response
            print(f"\n{'='*60}")
            print(f"📥 COGNITIVE CODE RESPONSE (Retry {retry + 1}/{max_retries})")
            print(f"{'='*60}")
            print(f"Raw response length: {len(code)} chars")
            
            if code:
                print(f"\nGenerated code preview (first 300 chars):")
                print(code[:300] + "..." if len(code) > 300 else code)
                print(f"{'='*60}\n")
                
                # Clean up code formatting
                code = clean_code_response(code)
                
                # Validate it's not empty after cleaning
                if code and len(code) > 20:
                    return code
                else:
                    print(f"⚠️ Code too short after cleaning, retrying...")
            else:
                print(f"\n⚠️ EMPTY RESPONSE FROM LLM")
                print(f"{'='*60}\n")
            
            # Exponential backoff
            if retry < max_retries - 1:
                wait_time = 2 ** retry
                print(f"⏳ Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
        
        except Exception as e:
            print(f"❌ Error generating code: {e}")
            if retry < max_retries - 1:
                wait_time = 2 ** retry
                print(f"⏳ Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                raise
    
    # If all retries failed, raise error
    raise RuntimeError(f"Failed to generate code after {max_retries} attempts")


def build_first_attempt_prompt(
    persona: SyntheticPersona,
    problem: Dict[str, Any],
    error_prediction: Dict[str, Any]
) -> str:
    """
    Build prompt for first attempt with error injection instructions.
    """
    
    # Extract problem details
    problem_statement = problem.get('question', problem.get('problem_statement_text', ''))
    examples = problem.get('examples', [])
    test_cases = problem.get('sample_test_cases', [])
    constraints = problem.get('constraints', '')
    
    # Build problem description
    problem_desc = f"# Problem\n{problem_statement}\n\n"
    
    if examples:
        problem_desc += "## Examples\n"
        for i, ex in enumerate(examples[:2], 1):
            problem_desc += f"Example {i}:\n"
            problem_desc += f"Input: {ex.get('input', '')}\n"
            problem_desc += f"Output: {ex.get('output', '')}\n\n"
    
    if test_cases:
        problem_desc += "## Test Cases\n"
        for tc in test_cases[:2]:
            problem_desc += f"Input: {tc.get('input', '')}\n"
            problem_desc += f"Expected: {tc.get('expected_output', '')}\n\n"
    
    if constraints:
        problem_desc += f"## Constraints\n{constraints}\n\n"
    
    # Build skill-level personality
    skill_personality = get_skill_personality(persona.skill_level)
    
    # Build error injection instructions
    if error_prediction['will_make_error']:
        error_instructions = f"""
CRITICAL: You MUST introduce these realistic student mistakes in your code:

Mistake Types to Include:
{chr(10).join(f"- {error}" for error in error_prediction['error_types'])}

Specific Mistakes to Inject (pick 1-2):
{chr(10).join(f"- {mistake}" for mistake in error_prediction['suggested_mistakes'])}

Weak Topics (expect mistakes here):
{chr(10).join(f"- {topic}" for topic in error_prediction['affected_topics']) if error_prediction['affected_topics'] else "- General implementation"}

Your code should ATTEMPT to solve the problem but MUST contain these realistic mistakes.
This simulates a real student's first attempt."""
    else:
        # Low error probability - allow mostly correct code with minor issues
        error_instructions = """
Your understanding is relatively good for this problem.
Write mostly correct code, but you may have 1 minor issue or suboptimal approach.
This represents a strong initial attempt."""
    
    # Build complete prompt
    prompt = f"""You are simulating a {persona.skill_level.upper()} student programmer.

STUDENT COGNITIVE PROFILE:
- Skill Level: {persona.skill_level}
- Topic Mastery: {error_prediction['avg_topic_mastery']:.1%}
- Working Memory: {persona.working_memory:.2f}
- Attention Span: {persona.attention_span:.2f}
- Metacognition: {persona.metacognition:.2f}

BEHAVIORAL CHARACTERISTICS:
{skill_personality}

{problem_desc}

{error_instructions}

CRITICAL CODE FORMAT:
Your code MUST follow this exact structure:

class Solution:
    def solution(self, param1, param2, ...):
        # Your implementation here
        # IMPORTANT: Always return the result
        # Even for in-place modifications, return the modified array/list
        return result

INSTRUCTIONS:
1. Write code that ATTEMPTS to solve the problem
2. Code MUST be in Solution class with solution method
3. Method MUST have 'self' as first parameter
4. Infer correct parameter names/types from examples
5. ALWAYS return the result (even for in-place operations)
6. Include the specified mistakes naturally in your implementation
7. Write ONLY executable Python code with MINIMAL or NO comments
8. DO NOT explain your mistakes in comments - write buggy code naturally
9. Keep code concise - no verbose explanations

Generate the Solution class code now (with the specified mistakes):"""
    
    return prompt


def build_retry_attempt_prompt(
    persona: SyntheticPersona,
    problem: Dict[str, Any],
    error_prediction: Dict[str, Any],
    previous_code: str,
    previous_errors: str,
    previous_hint: Optional[str],
    attempt_number: int
) -> str:
    """
    Build prompt for retry attempt showing learning progression.
    
    Key aspects:
    - Shows previous code and errors
    - Incorporates hint if provided
    - May still make mistakes (realistic learning)
    - Shows improvement over attempts
    """
    
    # Extract problem details
    problem_statement = problem.get('question', problem.get('problem_statement_text', ''))
    
    # Build skill-level personality
    skill_personality = get_skill_personality(persona.skill_level)
    
    # Determine if student should still make mistakes
    should_error = error_prediction['will_make_error']
    
    # Build hint incorporation instructions
    if previous_hint:
        hint_instructions = f"""
## Hint Received from Tutor
{previous_hint}

LEARNING INSTRUCTIONS:
- Read and try to understand the hint
- Incorporate the hint's suggestions into your solution
- Your hint_benefit factor is {persona.hint_benefit:.2f}
  * If high (>0.7): You understand hints well, incorporate thoroughly
  * If medium (0.4-0.7): You partially understand, incorporate some parts
  * If low (<0.4): You struggle to apply hints, may misinterpret

Based on your hint_benefit, incorporate the hint {'fully' if persona.hint_benefit > 0.7 else 'partially' if persona.hint_benefit > 0.4 else 'minimally'}.
"""
    else:
        hint_instructions = """
NO HINT RECEIVED:
Try to fix the error yourself based on the error message."""
    
    # Build error instructions for retry
    if should_error and should_make_mistake_on_retry(persona, attempt_number, previous_hint is not None):
        retry_error_instructions = f"""
REALISTIC LEARNING:
Even with the hint/error, you may still make mistakes:
- You might fix one issue but introduce another
- You might misunderstand the hint
- You might not fully address the edge cases

Possible continued mistakes (choose 0-1 if applicable):
{chr(10).join(f"- {mistake}" for mistake in error_prediction['suggested_mistakes'][:2])}
"""
    else:
        retry_error_instructions = """
IMPROVEMENT:
You're understanding the problem better now.
Write improved code that fixes the previous errors.
Your solution should be mostly or fully correct."""
    
    prompt = f"""You are simulating a {persona.skill_level.upper()} student programmer on attempt #{attempt_number}.

STUDENT COGNITIVE PROFILE:
- Skill Level: {persona.skill_level}
- Topic Mastery: {error_prediction['avg_topic_mastery']:.1%}
- Hint Benefit: {persona.hint_benefit:.2f}
- Metacognition: {persona.metacognition:.2f}

BEHAVIORAL CHARACTERISTICS:
{skill_personality}

# Original Problem
{problem_statement}

## Your Previous Attempt (Attempt #{attempt_number - 1})
```python
{previous_code}
```

## Result from Previous Attempt
{previous_errors}

{hint_instructions}

{retry_error_instructions}

CRITICAL CODE FORMAT:
class Solution:
    def solution(self, param1, param2, ...):
        # Fixed/improved implementation
        # IMPORTANT: Always return the result
        return result

INSTRUCTIONS:
1. Fix the errors from your previous attempt
2. Incorporate the hint (if provided) based on your hint_benefit
3. Keep Solution class structure with solution method
4. Method MUST have 'self' as first parameter
5. ALWAYS return the result (even for in-place operations)
6. Show learning progression from previous attempt
7. Write ONLY the corrected Python code with MINIMAL or NO comments
8. DO NOT add verbose explanations in comments - keep code concise
9. Focus on fixing the bug, not explaining it

Generate the improved Solution class code now:"""
    
    return prompt


def get_skill_personality(skill_level: str) -> str:
    """
    Get behavioral characteristics description for each skill level.
    """
    
    personalities = {
        'beginner': """- You often make syntax errors (missing colons, wrong indentation)
- You forget to handle edge cases (empty arrays, None values)
- You use inefficient approaches (nested loops when not needed)
- You struggle with complex data structures
- You make off-by-one errors in loops
- Your variable names may be unclear
- You sometimes have incomplete logic""",
        
        'intermediate': """- You usually have correct syntax
- You sometimes miss edge cases
- You understand basic algorithms but may choose suboptimal ones
- You occasionally make logical errors
- You might over-complicate simple problems
- You're learning to optimize but don't always succeed
- You understand hints but may apply them imperfectly""",
        
        'advanced': """- Your syntax is mostly correct
- You understand algorithm complexity
- You usually handle edge cases
- You may still make subtle logical errors
- You might miss an optimization opportunity
- You generally write clean, working code
- When you make errors, they're usually subtle
- You incorporate hints effectively"""
    }
    
    return personalities.get(skill_level, personalities['intermediate'])


def clean_code_response(code: str) -> str:
    """
    Clean LLM response to extract just the Python code.
    """
    
    # Remove markdown code blocks
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        # Generic code block
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1].strip()
    
    # Remove any leading/trailing whitespace
    code = code.strip()
    
    return code

