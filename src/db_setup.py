"""
Database Setup - Create and manage synthetic users in ArangoDB.

This module handles:
1. Creating synthetic users in IntelliT's ArangoDB users collection
2. Initializing learner_state with persona's initial_mastery
3. Cleanup of synthetic users after evaluation (optional)

Integration with:
- backend/app/db/database.py
- backend/app/crud/user.py
- backend/app/models/user.py
- backend/app/models/learner_state.py
"""

import sys
import os
from typing import List, Optional
from datetime import datetime, date

# Add backend to path for imports
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend'))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.db.database import get_db
from app.crud.user import UserCRUD
from app.models.user import UserCreate
from app.models.learner_state import LearnerState
from app.core.security import get_password_hash

from .persona_generator import SyntheticPersona


def create_synthetic_users(personas: List[SyntheticPersona]) -> List[str]:
    """
    Create users in ArangoDB users collection for all personas.
    
    Initializes learner_state with initial_mastery from persona.
    Sets is_synthetic flag for easy cleanup.
    
    Args:
        personas: List of SyntheticPersona objects
    
    Returns:
        List of created user._key values
    
    Raises:
        Exception: If database operation fails
    """
    try:
        db = get_db()
        user_crud = UserCRUD(db)
        created_keys = []
        
        print(f"\n{'='*60}")
        print(f"Creating {len(personas)} synthetic users in database")
        print(f"{'='*60}\n")
        
        for i, persona in enumerate(personas, 1):
            try:
                # Check if user already exists
                existing_user = user_crud.get_user_by_email(
                    f"{persona.user_key}@synth.intellit.ai"
                )
                
                if existing_user:
                    print(f"⚠️ [{i}/{len(personas)}] User {persona.user_key} already exists, skipping")
                    persona.database_user_key = existing_user.key
                    created_keys.append(existing_user.key)
                    continue
                
                # Create learner state with initial mastery
                learner_state = LearnerState(
                    version="1.0",
                    updated=datetime.utcnow(),
                    mastery=persona.initial_mastery,
                    reviews=[],
                    streak=0,
                    last_seen=date.today()
                )
                
                # Create user
                user_data = UserCreate(
                    email=f"{persona.user_key}@synth.intellit.ai",
                    password="synthetic_user_password_2024",  # Will be hashed
                    name=f"Synthetic {persona.skill_level.title()} #{persona.user_key[-3:]}",
                    skill_level=persona.skill_level
                )
                
                # Create user in database
                user = user_crud.create_user(user_data)
                
                # Update user with learner_state and synthetic flag
                user_crud.update_user_fields(user.key, {
                    "learner_state": learner_state.model_dump(mode='json'),
                    "is_synthetic": True,  # Flag for cleanup
                    "synthetic_metadata": {
                        "created_at": datetime.utcnow().isoformat(),
                        "learning_rate": persona.learning_rate,
                        "consistency": persona.consistency,
                        "hint_reliance": persona.hint_reliance,
                        "working_memory": persona.working_memory,
                        "metacognition": persona.metacognition
                    }
                })
                
                # Store database key in persona
                persona.database_user_key = user.key
                created_keys.append(user.key)
                
                print(f"✅ [{i}/{len(personas)}] Created {persona.user_key} (DB key: {user.key})")
                
            except Exception as e:
                print(f"❌ [{i}/{len(personas)}] Failed to create {persona.user_key}: {e}")
                raise
        
        print(f"\n{'='*60}")
        print(f"✅ Successfully created {len(created_keys)} synthetic users")
        print(f"{'='*60}\n")
        
        return created_keys
        
    except Exception as e:
        print(f"\n❌ Fatal error creating synthetic users: {e}")
        import traceback
        traceback.print_exc()
        raise


def update_synthetic_user_state(
    persona: SyntheticPersona,
    learner_state: LearnerState,
    additional_data: Optional[dict] = None
) -> bool:
    """
    Update synthetic user's learner state in database.
    
    Args:
        persona: SyntheticPersona with database_user_key set
        learner_state: Updated LearnerState object
        additional_data: Optional dict of additional fields to update
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if not persona.database_user_key:
            raise ValueError(f"Persona {persona.user_key} has no database_user_key")
        
        db = get_db()
        user_crud = UserCRUD(db)
        
        # Prepare updates
        updates = {
            "learner_state": learner_state.model_dump(mode='json'),
            "updated_at": datetime.utcnow()
        }
        
        # Add any additional data
        if additional_data:
            updates.update(additional_data)
        
        # Update in database
        user_crud.update_user_fields(persona.database_user_key, updates)
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating user state for {persona.user_key}: {e}")
        return False


def cleanup_synthetic_users(
    confirm: bool = False,
    keep_submissions: bool = True
) -> int:
    """
    Remove all synthetic users from database.
    
    WARNING: This will delete all users with is_synthetic=True flag.
    Use with caution!
    
    Args:
        confirm: Must be True to actually delete (safety check)
        keep_submissions: If True, keep submission records (default: True)
    
    Returns:
        Number of users deleted
    
    Raises:
        ValueError: If confirm is not True
    """
    if not confirm:
        raise ValueError(
            "cleanup_synthetic_users requires confirm=True to prevent accidental deletion"
        )
    
    try:
        db = get_db()
        
        # Find all synthetic users
        query = """
        FOR u IN users
        FILTER u.is_synthetic == true
        RETURN u
        """
        
        cursor = db.aql.execute(query)
        synthetic_users = list(cursor)
        
        if not synthetic_users:
            print("ℹ️ No synthetic users found to clean up")
            return 0
        
        print(f"\n{'='*60}")
        print(f"Cleaning up {len(synthetic_users)} synthetic users")
        print(f"{'='*60}\n")
        
        deleted_count = 0
        
        for user in synthetic_users:
            user_key = user['_key']
            
            try:
                # Optionally delete submissions
                if not keep_submissions:
                    delete_submissions_query = """
                    FOR s IN submissions
                    FILTER s.user_key == @user_key
                    REMOVE s IN submissions
                    """
                    db.aql.execute(delete_submissions_query, bind_vars={'user_key': user_key})
                
                # Delete user
                db.collection('users').delete(user_key)
                deleted_count += 1
                print(f"✅ Deleted {user.get('name', user_key)}")
                
            except Exception as e:
                print(f"❌ Failed to delete {user_key}: {e}")
        
        print(f"\n{'='*60}")
        print(f"✅ Cleaned up {deleted_count} synthetic users")
        if keep_submissions:
            print("ℹ️ Submissions were kept (keep_submissions=True)")
        print(f"{'='*60}\n")
        
        return deleted_count
        
    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_synthetic_user_count() -> int:
    """
    Get count of synthetic users in database.
    
    Returns:
        Number of users with is_synthetic=True
    """
    try:
        db = get_db()
        
        query = """
        FOR u IN users
        FILTER u.is_synthetic == true
        COLLECT WITH COUNT INTO count
        RETURN count
        """
        
        cursor = db.aql.execute(query)
        result = list(cursor)
        
        return result[0] if result else 0
        
    except Exception as e:
        print(f"❌ Error counting synthetic users: {e}")
        return 0


def verify_synthetic_users(personas: List[SyntheticPersona]) -> dict:
    """
    Verify that all personas have corresponding database users.
    
    Args:
        personas: List of SyntheticPersona objects
    
    Returns:
        Dict with verification results
    """
    try:
        db = get_db()
        user_crud = UserCRUD(db)
        
        results = {
            "total_personas": len(personas),
            "users_found": 0,
            "users_missing": 0,
            "missing_keys": []
        }
        
        for persona in personas:
            if not persona.database_user_key:
                results["users_missing"] += 1
                results["missing_keys"].append(persona.user_key)
                continue
            
            # Verify user exists
            user = user_crud.get_user_by_key(persona.database_user_key)
            if user:
                results["users_found"] += 1
            else:
                results["users_missing"] += 1
                results["missing_keys"].append(persona.user_key)
        
        print(f"\n{'='*60}")
        print(f"Synthetic User Verification")
        print(f"{'='*60}")
        print(f"Total Personas: {results['total_personas']}")
        print(f"Users Found: {results['users_found']}")
        print(f"Users Missing: {results['users_missing']}")
        
        if results["missing_keys"]:
            print(f"\nMissing Users:")
            for key in results["missing_keys"]:
                print(f"  - {key}")
        
        print(f"{'='*60}\n")
        
        return results
        
    except Exception as e:
        print(f"❌ Error verifying users: {e}")
        raise


def load_personas_from_database() -> List[SyntheticPersona]:
    """
    Load existing synthetic personas from database.
    
    Reconstructs SyntheticPersona objects from database records.
    Useful for resuming simulations or analysis.
    
    Returns:
        List of SyntheticPersona objects
    """
    try:
        db = get_db()
        
        query = """
        FOR u IN users
        FILTER u.is_synthetic == true
        RETURN u
        """
        
        cursor = db.aql.execute(query)
        users = list(cursor)
        
        personas = []
        
        for user in users:
            # Extract synthetic metadata
            metadata = user.get('synthetic_metadata', {})
            learner_state = user.get('learner_state', {})
            
            # Reconstruct persona
            persona = SyntheticPersona(
                user_key=user['email'].split('@')[0],  # Extract from email
                skill_level=user.get('skill_level', 'intermediate'),
                initial_mastery=learner_state.get('mastery', {}),
                learning_rate=metadata.get('learning_rate', 0.12),
                mistake_patterns=[],  # Not stored in DB
                working_memory=metadata.get('working_memory', 0.7),
                metacognition=metadata.get('metacognition', 0.5),
                consistency=metadata.get('consistency', 0.8),
                hint_reliance=metadata.get('hint_reliance', 0.5),
                session_duration_minutes=60,
                fatigue_rate=0.15,
                database_user_key=user['_key'],
                current_mastery=learner_state.get('mastery', {}).copy()
            )
            
            personas.append(persona)
        
        print(f"✅ Loaded {len(personas)} synthetic personas from database")
        
        return personas
        
    except Exception as e:
        print(f"❌ Error loading personas from database: {e}")
        import traceback
        traceback.print_exc()
        raise

