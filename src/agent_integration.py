"""
IntelliT API Client - Make API calls like a headless user.

This module uses the backend REST API properly:
1. Login to get JWT tokens
2. Create sessions for each problem
3. Submit code via API (automatic learner state updates)
4. Request hints via API (automatic session tracking)

This approach ensures ALL metrics are tracked correctly:
- Session analytics (hints_used, tests_run, etc.)
- Learner state updates (mastery, reviews)
- Submission records
- Behavior tracking

NO direct internal calls - everything through the API like a real user.

LLM Configuration (matches backend):
- Feedback Agent: Gemini 2.5 Flash, 2048 max_output_tokens, temp 0.7
- Code Analysis Agent: Gemini 2.5 Flash, 4096 max_output_tokens, temp 0.4
- Gemini supports up to 65,535 output tokens (plenty of headroom)
- Adaptive thinking uses internal reasoning budget automatically
"""

import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from .persona_generator import SyntheticPersona, determine_hint_level


# Backend API configuration
API_BASE_URL = "http://localhost:8000"

# API Timeouts - Backend runs locally with Gemini 2.5 Flash
# Gemini is fast but we account for complex problems and hints
API_TIMEOUTS = {
    "connect": 10,      # Connection establishment
    "session_ops": 30,  # Session start/end
    "submission": 180,  # 3 minutes for code execution + learner state updates
    "hint": 180,        # 3 minutes for LLM generation (levels 1-5 can be complex)
    "total": 300        # 5 minutes max per operation
}


class IntelliTAPIClient:
    """
    API client for synthetic users to interact with IntelliT backend.
    
    Handles authentication, session management, and all API calls.
    """
    
    def __init__(self, persona: SyntheticPersona, password: str = "synthetic_user_password_2024"):
        self.persona = persona
        self.email = f"{persona.user_key}@synth.intellit.ai"
        self.password = password
        self.access_token = None
        self.current_session_id = None
    
    async def login(self) -> bool:
        """
        Login and get JWT access token.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            timeout = aiohttp.ClientTimeout(
                total=API_TIMEOUTS["session_ops"],
                connect=API_TIMEOUTS["connect"]
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{API_BASE_URL}/auth/login"
                payload = {
                    "email": self.email,
                    "password": self.password
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.access_token = data["access_token"]
                        print(f"✅ Logged in: {self.persona.user_key}")
                        return True
                    else:
                        error = await response.text()
                        print(f"❌ Login failed for {self.persona.user_key}: {error}")
                        return False
                        
        except Exception as e:
            print(f"❌ Login exception for {self.persona.user_key}: {e}")
            return False
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers with JWT token."""
        if not self.access_token:
            raise ValueError(f"Not logged in: {self.persona.user_key}")
        
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    async def start_session(self, problem: Dict[str, Any]) -> Optional[str]:
        """
        Start a coding session for a problem.
        
        Args:
            problem: Question dict from intellit_adapter
        
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            timeout = aiohttp.ClientTimeout(
                total=API_TIMEOUTS["session_ops"],
                connect=API_TIMEOUTS["connect"]
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{API_BASE_URL}/sessions/start"
                headers = self.get_auth_headers()
                
                payload = {
                    "session_type": "practice",  # Lowercase per SessionType enum
                    "question_id": problem['question_id'],
                    "question_title": problem.get('title', 'Untitled'),
                    "roadmap_id": "strivers-a2z",
                    "difficulty": problem.get('difficulty', 'Medium'),
                    "programming_language": "python",
                    "config": {
                        "enable_behavior_tracking": False  # Disable for synthetic users
                    }
                }
                
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        # SessionResponse has session_id at top level, not nested
                        self.current_session_id = data["session_id"]
                        print(f"  📋 Session started: {self.current_session_id}")
                        return self.current_session_id
                    else:
                        error = await response.text()
                        print(f"  ⚠️ Session start failed: {error}")
                        return None
                        
        except Exception as e:
            print(f"  ❌ Session start exception: {e}")
            return None
    
    async def submit_code(
        self,
        problem: Dict[str, Any],
        code: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Submit code via API endpoint.
        
        This automatically:
        - Executes code with test cases
        - Creates submission record
        - Updates learner state
        - Updates session analytics
        - Triggers code analysis agent if all tests pass
        
        Args:
            problem: Question dict
            code: User's solution code
            language: Programming language
        
        Returns:
            Dict with execution results
        """
        try:
            # Longer timeout for submissions: code execution + DB updates + optional analysis
            timeout = aiohttp.ClientTimeout(
                total=API_TIMEOUTS["submission"],
                connect=API_TIMEOUTS["connect"]
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{API_BASE_URL}/submissions/submit"
                headers = self.get_auth_headers()
                
                # Format test cases
                test_cases = [
                    {
                        "input": tc.get('input', ''),
                        "expected_output": tc.get('expected_output', '')
                    }
                    for tc in problem.get('sample_test_cases', [])
                ]
                
                payload = {
                    "code": code,
                    "language": language,
                    "question_key": problem['question_id'],
                    "question_title": problem.get('title', 'Untitled'),
                    "test_cases": test_cases,
                    "session_id": self.current_session_id,
                    "roadmap_id": "strivers-a2z",
                    "difficulty": problem.get('difficulty', 'Medium'),
                    "function_name": None
                }
                
                # Log submission payload for debugging
                print(f"\n{'='*60}")
                print(f"📤 CODE SUBMISSION TO BACKEND")
                print(f"{'='*60}")
                print(f"Question: {payload['question_title']}")
                print(f"Language: {payload['language']}")
                print(f"Test Cases: {len(test_cases)}")
                print(f"\nCode being submitted:")
                print(code)
                print(f"{'='*60}\n")
                
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract key info
                        result = {
                            "success": data.get("success", False),
                            "status": data.get("status", "Unknown"),
                            "submission_id": data.get("submission_id"),
                            "passed_count": data.get("passed_count", 0),
                            "total_count": data.get("total_count", 0),
                            "runtime_ms": data.get("runtime_ms"),
                            "memory_kb": data.get("memory_kb"),
                            "error_message": data.get("error_message"),
                            "test_results": data.get("test_results", [])
                        }
                        
                        print(f"  📊 Submission: {result['status']} ({result['passed_count']}/{result['total_count']})")
                        
                        # Detailed error logging for failures
                        if not result['success']:
                            print(f"\n{'='*60}")
                            print(f"❌ SUBMISSION FAILED - DETAILED ANALYSIS")
                            print(f"{'='*60}")
                            print(f"Status: {result['status']}")
                            print(f"Passed: {result['passed_count']}/{result['total_count']} test cases")
                            
                            if result['error_message']:
                                print(f"\nError Message:")
                                print(result['error_message'])
                            
                            # Show test case details
                            if result['test_results']:
                                print(f"\nTest Case Results:")
                                for i, test_result in enumerate(result['test_results'][:3], 1):  # Show first 3
                                    status = "✅ PASS" if test_result.get('passed') else "❌ FAIL"
                                    print(f"\n  Test {i}: {status}")
                                    print(f"    Input: {test_result.get('input', 'N/A')}")
                                    print(f"    Expected: {test_result.get('expected_output', 'N/A')}")
                                    print(f"    Actual: {test_result.get('actual_output', 'None')}")
                                    if test_result.get('error'):
                                        print(f"    Error: {test_result.get('error')}")
                            
                            print(f"{'='*60}\n")
                        
                        return result
                    else:
                        error_text = await response.text()
                        print(f"  ❌ Submission failed: {error_text}")
                        return {
                            "success": False,
                            "status": "API Error",
                            "passed_count": 0,
                            "total_count": len(test_cases),
                            "error_message": error_text
                        }
                        
        except Exception as e:
            print(f"  ❌ Submission exception: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "status": "Exception",
                "passed_count": 0,
                "total_count": len(problem.get('sample_test_cases', [])),
                "error_message": str(e)
            }
    
    async def request_hint(
        self,
        problem: Dict[str, Any],
        code: str,
        hint_level: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Request hint via API endpoint.
        
        This automatically:
        - Calculates user proficiency (5 weighted metrics)
        - Auto-increments hint level based on session
        - Enforces 5-hint limit per session
        - Includes last_run test failure context
        - Generates graduated pedagogical hints via Gemini
        - Updates session analytics
        
        Backend uses:
        - Gemini 2.5 Flash with 2048 max_output_tokens
        - Adaptive hints based on proficiency (beginner/intermediate/advanced)
        - 5 hint levels: Metacognitive → Conceptual → Strategic → Structural → Targeted
        
        Args:
            problem: Question dict
            code: Current code attempt
            hint_level: Hint level (1-5), auto-calculated if None
        
        Returns:
            Dict with hint data
        """
        try:
            # Determine hint level if not provided
            if hint_level is None:
                attempt_number = 1  # Will be auto-calculated by backend from session
                hint_level = determine_hint_level(self.persona, attempt_number)
            
            # Longer timeout for hints: proficiency calculation + LLM generation
            timeout = aiohttp.ClientTimeout(
                total=API_TIMEOUTS["hint"],
                connect=API_TIMEOUTS["connect"]
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{API_BASE_URL}/agents/hint"
                headers = self.get_auth_headers()
                
                payload = {
                    "question_id": problem['question_id'],
                    "code": code or "",
                    "hint_level": hint_level,
                    "session_id": self.current_session_id
                }
                
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        hint_result = {
                            "success": True,
                            "hint_text": data.get("hint_text", ""),
                            "hint_level": data.get("hint_level", hint_level),
                            "level_name": data.get("level_name", ""),
                            "hints_used_total": data.get("hints_used_total", 0),
                            "hints_remaining": data.get("hints_remaining", 0)
                        }
                        
                        print(f"  💡 Hint received: Level {hint_result['hint_level']} ({hint_result['hints_used_total']}/5 used)")
                        return hint_result
                    else:
                        error_text = await response.text()
                        print(f"  ⚠️ Hint request failed: {error_text}")
                        return {
                            "success": False,
                            "hint_text": "Hint unavailable",
                            "hint_level": hint_level,
                            "level_name": "Error",
                            "error": error_text
                        }
                        
        except Exception as e:
            print(f"  ❌ Hint request exception: {e}")
            return {
                "success": False,
                "hint_text": "Error requesting hint",
                "hint_level": hint_level or 1,
                "level_name": "Error",
                "error": str(e)
            }
    
    async def end_session(self) -> bool:
        """
        End the current session.
        
        Returns:
            True if successful
        """
        if not self.current_session_id:
            return True
        
        try:
            timeout = aiohttp.ClientTimeout(
                total=API_TIMEOUTS["session_ops"],
                connect=API_TIMEOUTS["connect"]
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Correct endpoint: /{session_id}/end with optional reason param
                url = f"{API_BASE_URL}/sessions/{self.current_session_id}/end"
                headers = self.get_auth_headers()
                
                # Reason is optional query param, defaults to "user_request"
                params = {"reason": "simulation_completed"}
                
                async with session.post(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        print(f"  ✅ Session ended: {self.current_session_id}")
                        self.current_session_id = None
                        return True
                    else:
                        print(f"  ⚠️ Session end warning: {response.status}")
                        self.current_session_id = None
                        return False
                        
        except Exception as e:
            print(f"  ⚠️ Session end exception: {e}")
            self.current_session_id = None
            return False


# ============================================================================
# Helper Functions for Simulation
# ============================================================================

async def simulate_hint_interaction(
    api_client: IntelliTAPIClient,
    problem: Dict[str, Any],
    code: str,
    max_hints: int = 3
) -> List[Dict[str, Any]]:
    """
    Simulate a full hint interaction sequence.
    
    Requests multiple hints if persona continues to struggle,
    with automatic escalation via backend.
    
    Args:
        api_client: Authenticated API client
        problem: Question dict
        code: Current code attempt
        max_hints: Maximum hints to request (default: 3)
    
    Returns:
        List of hint dicts received
    """
    hints_received = []
    
    for attempt in range(1, max_hints + 1):
        # Check if persona wants another hint
        from .persona_generator import should_request_hint
        if not should_request_hint(api_client.persona, attempt):
            break
        
        # Request hint (backend auto-increments level)
        hint = await api_client.request_hint(
            problem=problem,
            code=code,
            hint_level=None  # Let backend calculate
        )
        
        if hint.get("success"):
            hints_received.append(hint)
            api_client.persona.hints_requested += 1
        else:
            # Stop if hint fails or limit reached
            break
        
        # Small delay between hint requests
        await asyncio.sleep(0.1)
    
    return hints_received


def calculate_hint_effectiveness(
    hints_received: List[Dict[str, Any]],
    next_attempt_success: bool
) -> Dict[str, Any]:
    """
    Calculate effectiveness metrics for hints.
    
    Args:
        hints_received: List of hint dicts
        next_attempt_success: Whether next attempt after hints was successful
    
    Returns:
        Dict with effectiveness metrics
    """
    if not hints_received:
        return {
            "hints_count": 0,
            "helped": False,
            "success_after_hints": False
        }
    
    return {
        "hints_count": len(hints_received),
        "max_level_reached": max(h.get("hint_level", 1) for h in hints_received),
        "all_successful": all(h.get("success", False) for h in hints_received),
        "helped": next_attempt_success,
        "success_after_hints": next_attempt_success
    }


async def test_api_integration(persona: SyntheticPersona, problem: Dict[str, Any]):
    """
    Test API integration for a single persona and problem.
    
    Useful for debugging and validation.
    
    Args:
        persona: SyntheticPersona (must have database_user_key set)
        problem: Question dict from intellit_adapter
    """
    print(f"\n{'='*60}")
    print(f"Testing API Integration")
    print(f"Persona: {persona.user_key}")
    print(f"Problem: {problem.get('title', 'Untitled')}")
    print(f"{'='*60}\n")
    
    # Create API client
    api_client = IntelliTAPIClient(persona)
    
    # Test login
    print("1. Testing Login...")
    success = await api_client.login()
    if not success:
        print("❌ Login failed, aborting test")
        return
    
    # Test session creation
    print("\n2. Testing Session Creation...")
    session_id = await api_client.start_session(problem)
    if not session_id:
        print("❌ Session creation failed, aborting test")
        return
    
    # Test code submission
    print("\n3. Testing Code Submission...")
    dummy_code = "def solution(nums):\n    return sum(nums)"
    result = await api_client.submit_code(problem, dummy_code)
    print(f"   Success: {result.get('success')}")
    print(f"   Status: {result.get('status')}")
    
    # Test hint request
    print("\n4. Testing Hint Request...")
    hint = await api_client.request_hint(problem, dummy_code)
    print(f"   Success: {hint.get('success')}")
    print(f"   Hint Level: {hint.get('hint_level')}")
    print(f"   Hint (preview): {hint.get('hint_text', '')[:100]}...")
    
    # Test session end
    print("\n5. Testing Session End...")
    await api_client.end_session()
    
    print(f"\n{'='*60}")
    print("✅ API Integration Test Complete")
    print(f"{'='*60}\n")
