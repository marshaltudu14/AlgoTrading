"""
Centralized error and warning logging system with deduplication.
Maintains a clean log file with only unique warnings and errors.
"""

import os
import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set
from collections import defaultdict

class ErrorLogger:
    """Centralized error and warning logger with smart deduplication."""

    def __init__(self, log_file: str = "training_errors.txt"):
        self.log_file = log_file
        self.errors = []
        self.warnings = []
        self.session_start = datetime.now()

        # Deduplication tracking
        self.message_counts = defaultdict(int)
        self.message_timestamps = {}
        self.logged_messages = set()
        self.dedup_window = timedelta(minutes=5)  # Don't repeat same message within 5 minutes

        # Clear previous log file
        self._clear_log_file()

        # Set up file handler for errors/warnings only
        self.file_handler = logging.FileHandler(self.log_file, mode='w')
        self.file_handler.setLevel(logging.WARNING)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(self.file_handler)

        self._write_header()
    
    def _clear_log_file(self):
        """Clear the log file to start fresh."""
        if os.path.exists(self.log_file):
            try:
                os.remove(self.log_file)
            except PermissionError:
                # File is in use, try to truncate it instead
                try:
                    with open(self.log_file, 'w') as f:
                        f.write("")
                except Exception:
                    # If we can't clear it, just continue
                    pass
    
    def _write_header(self):
        """Write session header to log file."""
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRADING SYSTEM - ERRORS & WARNINGS LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Session started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def _get_message_key(self, message: str, context: str = None) -> str:
        """Generate a unique key for message deduplication."""
        # Create a simplified version of the message for deduplication
        # Remove timestamps, numbers, and other variable parts
        import re

        # Remove common variable parts
        clean_msg = re.sub(r'\d+\.\d+', 'X.X', message)  # Replace decimals
        clean_msg = re.sub(r'\d+', 'N', clean_msg)  # Replace integers
        clean_msg = re.sub(r'episode \d+', 'episode N', clean_msg, flags=re.IGNORECASE)
        clean_msg = re.sub(r'agent \d+', 'agent N', clean_msg, flags=re.IGNORECASE)
        clean_msg = re.sub(r'step \d+', 'step N', clean_msg, flags=re.IGNORECASE)

        # Include context in key if provided
        key_content = f"{clean_msg}|{context}" if context else clean_msg
        return hashlib.md5(key_content.encode()).hexdigest()[:12]

    def _should_log_message(self, message_key: str) -> bool:
        """Check if message should be logged based on deduplication rules."""
        now = datetime.now()

        # If we've never seen this message, log it
        if message_key not in self.message_timestamps:
            return True

        # If enough time has passed since last occurrence, log it
        last_time = self.message_timestamps[message_key]
        if now - last_time > self.dedup_window:
            return True

        return False

    def log_error(self, error_msg: str, context: str = None):
        """Log an error with optional context and deduplication."""
        message_key = self._get_message_key(error_msg, context)
        now = datetime.now()

        # Always track the message
        self.message_counts[message_key] += 1
        self.message_timestamps[message_key] = now

        # Only log if it should be logged (deduplication check)
        if self._should_log_message(message_key):
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

            error_entry = {
                'timestamp': timestamp,
                'type': 'ERROR',
                'message': error_msg,
                'context': context,
                'count': self.message_counts[message_key]
            }

            self.errors.append(error_entry)
            self._append_to_file(error_entry)

    def log_warning(self, warning_msg: str, context: str = None):
        """Log a warning with optional context and deduplication."""
        message_key = self._get_message_key(warning_msg, context)
        now = datetime.now()

        # Always track the message
        self.message_counts[message_key] += 1
        self.message_timestamps[message_key] = now

        # Only log if it should be logged (deduplication check)
        if self._should_log_message(message_key):
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

            warning_entry = {
                'timestamp': timestamp,
                'type': 'WARNING',
                'message': warning_msg,
                'context': context,
                'count': self.message_counts[message_key]
            }

            self.warnings.append(warning_entry)
            self._append_to_file(warning_entry)
    
    def _append_to_file(self, entry: Dict[str, Any]):
        """Append entry to log file with count information."""
        with open(self.log_file, 'a') as f:
            count_info = f" (x{entry['count']})" if entry['count'] > 1 else ""
            f.write(f"[{entry['timestamp']}] {entry['type']}: {entry['message']}{count_info}\n")
            if entry['context']:
                f.write(f"    Context: {entry['context']}\n")
            f.write("\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of errors and warnings."""
        return {
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'session_duration': datetime.now() - self.session_start,
            'recent_errors': self.errors[-5:] if self.errors else [],
            'recent_warnings': self.warnings[-5:] if self.warnings else []
        }
    
    def write_summary(self):
        """Write final summary to log file with deduplication stats."""
        summary = self.get_summary()

        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("SESSION SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Unique Errors Logged: {summary['total_errors']}\n")
            f.write(f"Unique Warnings Logged: {summary['total_warnings']}\n")
            f.write(f"Total Message Occurrences: {sum(self.message_counts.values())}\n")
            f.write(f"Unique Message Types: {len(self.message_counts)}\n")
            f.write(f"Session Duration: {summary['session_duration']}\n")
            f.write("=" * 80 + "\n")

            # Write top repeated messages
            if self.message_counts:
                f.write("\nMOST FREQUENT ISSUES:\n")
                f.write("-" * 40 + "\n")
                sorted_messages = sorted(self.message_counts.items(), key=lambda x: x[1], reverse=True)
                for i, (msg_key, count) in enumerate(sorted_messages[:5]):
                    f.write(f"{i+1}. Message pattern occurred {count} times\n")
                f.write("-" * 40 + "\n")

# Global error logger instance
error_logger = ErrorLogger()

def log_error(message: str, context: str = None):
    """Global function to log errors."""
    error_logger.log_error(message, context)

def log_warning(message: str, context: str = None):
    """Global function to log warnings."""
    error_logger.log_warning(message, context)

def get_error_summary():
    """Get error summary."""
    return error_logger.get_summary()
