"""
IntelliT Adapter - Bridge between IntelliT roadmap schema and student_sim format.

This module provides functions to:
1. Load questions from IntelliT's ArangoDB roadmap collection
2. Convert RoadmapItem to student_sim's expected data format
3. Create proper submission records from synthetic user attempts

Integration with:
- backend/app/models/roadmap.py
- backend/app/crud/roadmap.py
- backend/app/models/submission.py
"""

import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Add backend to path for imports
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend'))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.db.database import get_db
from app.models.roadmap import RoadmapItem
from app.models.submission import SubmissionCreate, SubmissionStatus
from app.crud.submission import SubmissionCRUD


def load_strivers_a2z_questions(limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Load questions from roadmap collection filtered by course='strivers-a2z'.
    
    Args:
        limit: Maximum number of questions to load
        offset: Number of questions to skip (for pagination)
    
    Returns:
        List of roadmap questions in student_sim compatible format
    
    Raises:
        Exception: If database connection fails
    """
    try:
        db = get_db()
        
        # Query roadmap collection for strivers-a2z course
        query = """
        FOR r IN roadmap
        FILTER r.course == @course
        FILTER r.scraping_success == true
        FILTER r.problem_statement_text != null
        FILTER r.sample_test_cases != null
        FILTER LENGTH(r.sample_test_cases) > 0
        SORT r.step_number ASC
        LIMIT @offset, @limit
        RETURN r
        """
        
        cursor = db.aql.execute(query, bind_vars={
            'course': 'strivers-a2z',
            'offset': offset,
            'limit': limit
        })
        
        questions = []
        for roadmap_data in cursor:
            # Convert to student_sim format
            question = convert_roadmap_to_student_sim_format(roadmap_data)
            if question:
                questions.append(question)
        
        print(f"✅ Loaded {len(questions)} questions from strivers-a2z (offset={offset}, limit={limit})")
        return questions
        
    except Exception as e:
        print(f"❌ Error loading questions from roadmap: {e}")
        import traceback
        traceback.print_exc()
        raise


def convert_roadmap_to_student_sim_format(roadmap_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert RoadmapItem to student_sim format.
    
    Student_sim expects:
    {
        'question_id': str,
        'question': str (problem statement),
        'desc': str (difficulty + topics),
        'topics': List[str],
        'sample_test_cases': List[Dict],
        'difficulty': str,
        'original_data': Dict (preserve original for reference)
    }
    
    Args:
        roadmap_item: Dictionary from roadmap collection
    
    Returns:
        Question in student_sim format, or None if invalid
    """
    try:
        # Validate required fields
        if not roadmap_item.get('problem_statement_text'):
            print(f"⚠️ Skipping {roadmap_item.get('_key')}: No problem statement")
            return None
        
        if not roadmap_item.get('sample_test_cases'):
            print(f"⚠️ Skipping {roadmap_item.get('_key')}: No test cases")
            return None
        
        # Extract topics from a2z_topics (JSON string) or topics list
        topics = []
        if roadmap_item.get('topics') and isinstance(roadmap_item['topics'], list):
            topics = roadmap_item['topics']
        elif roadmap_item.get('a2z_topics'):
            try:
                if isinstance(roadmap_item['a2z_topics'], str):
                    topics = json.loads(roadmap_item['a2z_topics'])
                else:
                    topics = roadmap_item['a2z_topics']
            except json.JSONDecodeError:
                topics = [roadmap_item['a2z_topics']]
        
        # Build description combining difficulty and topics
        difficulty = roadmap_item.get('leetcode_difficulty', 'Medium')
        topic_str = ', '.join(topics[:3]) if topics else 'General'
        desc = f"Difficulty: {difficulty}. Topics: {topic_str}"
        
        # Format test cases
        test_cases = []
        for tc in roadmap_item.get('sample_test_cases', []):
            test_cases.append({
                'input': tc.get('input', ''),
                'expected_output': tc.get('expected_output') or tc.get('output', '')
            })
        
        return {
            'question_id': roadmap_item.get('_key'),
            'leetcode_question_id': roadmap_item.get('leetcode_question_id'),
            'question': roadmap_item['problem_statement_text'],
            'desc': desc,
            'topics': topics,
            'sample_test_cases': test_cases,
            'difficulty': difficulty,
            'title': roadmap_item.get('leetcode_title') or roadmap_item.get('original_title', 'Untitled'),
            'hints': roadmap_item.get('hints', []),
            'step_number': roadmap_item.get('step_number', 0),
            'original_data': roadmap_item  # Preserve full original data
        }
        
    except Exception as e:
        print(f"❌ Error converting roadmap item {roadmap_item.get('_key')}: {e}")
        return None


def create_submission_from_synthetic(
    user_key: str,
    question: Dict[str, Any],
    code: str,
    execution_result: Dict[str, Any],
    language: str = "python",
    session_id: Optional[str] = None,
    hints_used: int = 0,
    time_taken_seconds: int = 0
) -> str:
    """
    Create proper IntelliT submission record from synthetic attempt.
    
    Args:
        user_key: Synthetic user's database key
        question: Question dict from convert_roadmap_to_student_sim_format
        code: User's solution code
        execution_result: Dict with status, passed_count, total_count, runtime_ms, error_message
        language: Programming language (default: python)
        session_id: Optional session ID for tracking
        hints_used: Number of hints user requested
        time_taken_seconds: Time spent on problem
    
    Returns:
        Submission document key (_key)
    
    Raises:
        Exception: If submission creation fails
    """
    try:
        db = get_db()
        submission_crud = SubmissionCRUD(db)
        
        # Map execution status to SubmissionStatus
        status_str = execution_result.get('status', 'Wrong Answer')
        if status_str == 'Accepted':
            status = SubmissionStatus.ACCEPTED
        elif 'Runtime Error' in status_str:
            status = SubmissionStatus.RUNTIME_ERROR
        elif 'Time Limit' in status_str:
            status = SubmissionStatus.TIME_LIMIT_EXCEEDED
        elif 'Compile Error' in status_str:
            status = SubmissionStatus.COMPILE_ERROR
        else:
            status = SubmissionStatus.WRONG_ANSWER
        
        # Extract performance metrics - ALL FROM REAL EXECUTION
        runtime_ms = execution_result.get('runtime_ms')
        memory_kb = execution_result.get('memory_kb')
        passed_count = execution_result.get('passed_count', 0)
        total_count = execution_result.get('total_count', len(question.get('sample_test_cases', [])))
        error_message = execution_result.get('error_message')
        
        # Percentiles: NO FAKE DATA
        # Backend calculates these from actual distribution of all submissions
        # For synthetic users, we don't fabricate percentiles
        runtime_percentile = execution_result.get('runtime_percentile')  # From backend if available
        memory_percentile = execution_result.get('memory_percentile')    # From backend if available
        
        # Create submission data
        submission_data = SubmissionCreate(
            question_key=question['question_id'],
            question_title=question.get('title', 'Untitled'),
            code=code,
            language=language,
            status=status,
            runtime_ms=runtime_ms,
            memory_kb=memory_kb,
            total_test_cases=total_count,
            passed_test_cases=passed_count,
            failed_test_case_index=execution_result.get('failed_test_case_index'),
            error_message=error_message,
            runtime_percentile=runtime_percentile,
            memory_percentile=memory_percentile,
            session_id=session_id,
            time_taken_seconds=time_taken_seconds,
            hints_used=hints_used,
            roadmap_id='strivers-a2z',
            difficulty=question.get('difficulty', 'Medium'),
            points_earned=0  # Points disabled for synthetic users
        )
        
        # Save submission
        submission = submission_crud.create_submission(user_key, submission_data)
        
        return submission.key
        
    except Exception as e:
        print(f"❌ Error creating submission: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_question_topics(question: Dict[str, Any]) -> List[str]:
    """
    Extract clean topic list from question dict.
    
    Args:
        question: Question dict from convert_roadmap_to_student_sim_format
    
    Returns:
        List of topic strings
    """
    return question.get('topics', [])


def filter_questions_by_difficulty(
    questions: List[Dict[str, Any]], 
    difficulty: str
) -> List[Dict[str, Any]]:
    """
    Filter questions by difficulty level.
    
    Args:
        questions: List of questions
        difficulty: "Easy", "Medium", or "Hard"
    
    Returns:
        Filtered list of questions
    """
    return [q for q in questions if q.get('difficulty') == difficulty]


def filter_questions_by_topics(
    questions: List[Dict[str, Any]], 
    topics: List[str]
) -> List[Dict[str, Any]]:
    """
    Filter questions that contain any of the specified topics.
    
    Args:
        questions: List of questions
        topics: List of topic strings to match
    
    Returns:
        Filtered list of questions
    """
    filtered = []
    for q in questions:
        q_topics = q.get('topics', [])
        if any(topic in q_topics for topic in topics):
            filtered.append(q)
    return filtered


def get_question_by_id(questions: List[Dict[str, Any]], question_id: str) -> Optional[Dict[str, Any]]:
    """
    Find question by ID in list.
    
    Args:
        questions: List of questions
        question_id: Question ID to find
    
    Returns:
        Question dict or None if not found
    """
    for q in questions:
        if q['question_id'] == question_id:
            return q
    return None

