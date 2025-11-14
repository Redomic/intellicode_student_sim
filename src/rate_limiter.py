"""
Rate Limiter for LLM API Calls

Prevents hitting API rate limits by throttling requests across ALL concurrent threads.
Uses token bucket algorithm for smooth rate limiting.

THREAD SAFETY:
- Single global instance shared across all async coroutines/threads
- asyncio.Lock ensures thread-safe request tracking
- All concurrent simulations respect the same rate limit

HARD LIMIT: 10 RPM (Requests Per Minute)
- Applies to ALL LLM calls: code generation, hints, analysis, orchestrator
- With 2 concurrent threads: each gets ~5 requests/min on average
- Automatic blocking if limit reached (async sleep)

USAGE:
    from src.rate_limiter import get_gemini_rate_limiter
    
    rate_limiter = get_gemini_rate_limiter()
    await rate_limiter.acquire()  # Blocks until request allowed
    # ... make API call ...
"""

import asyncio
import time
from typing import Optional
from collections import deque


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Ensures we don't exceed rate limits by controlling request frequency.
    """
    
    def __init__(
        self,
        max_requests_per_minute: int = 10,
        max_requests_per_hour: Optional[int] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_requests_per_minute: Maximum requests per minute
            max_requests_per_hour: Optional hourly limit
        """
        self.max_rpm = max_requests_per_minute
        self.max_rph = max_requests_per_hour
        
        # Track request timestamps
        self.minute_window = deque(maxlen=max_requests_per_minute)
        self.hour_window = deque() if max_requests_per_hour else None
        
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """
        Acquire permission to make a request.
        
        Blocks until rate limit allows the request.
        """
        async with self.lock:
            now = time.time()
            
            # Clean up old timestamps
            self._cleanup_windows(now)
            
            # Calculate wait time
            wait_time = self._calculate_wait_time(now)
            
            if wait_time > 0:
                # Show which limit was hit
                in_minute = len(self.minute_window)
                in_hour = len(self.hour_window) if self.hour_window else 0
                print(f"  ⏳ Rate limit: {in_minute}/{self.max_rpm} RPM, waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                now = time.time()
            
            # Record this request
            self.minute_window.append(now)
            if self.hour_window is not None:
                self.hour_window.append(now)
    
    def _cleanup_windows(self, now: float):
        """Remove timestamps outside the window."""
        # Clean minute window
        while self.minute_window and now - self.minute_window[0] > 60:
            self.minute_window.popleft()
        
        # Clean hour window
        if self.hour_window is not None:
            while self.hour_window and now - self.hour_window[0] > 3600:
                self.hour_window.popleft()
    
    def _calculate_wait_time(self, now: float) -> float:
        """Calculate how long to wait before next request."""
        wait_times = []
        
        # Check minute limit
        if len(self.minute_window) >= self.max_rpm:
            oldest = self.minute_window[0]
            wait_times.append(60 - (now - oldest))
        
        # Check hour limit
        if self.hour_window is not None and self.max_rph is not None:
            if len(self.hour_window) >= self.max_rph:
                oldest = self.hour_window[0]
                wait_times.append(3600 - (now - oldest))
        
        return max(wait_times) if wait_times else 0


# Global rate limiter for Gemini API
# HARD LIMIT: 10 RPM to prevent rate limit errors
# All concurrent threads share this single rate limiter instance
_gemini_limiter = RateLimiter(
    max_requests_per_minute=10,  # HARD LIMIT: 10 RPM max
    max_requests_per_hour=500    # Daily limit: ~1000 RPD (conservative)
)


def get_gemini_rate_limiter() -> RateLimiter:
    """Get global Gemini rate limiter instance."""
    return _gemini_limiter

