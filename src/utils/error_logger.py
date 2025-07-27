"""
Centralized error and warning logging system.
Maintains a clean log file with only the latest warnings and errors.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any

class ErrorLogger:
    """Centralized error and warning logger that maintains a clean log file."""
    
    def __init__(self, log_file: str = "training_errors.txt"):
        self.log_file = log_file
        self.errors = []
        self.warnings = []
        self.session_start = datetime.now()
        
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
            os.remove(self.log_file)
    
    def _write_header(self):
        """Write session header to log file."""
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRADING SYSTEM - ERRORS & WARNINGS LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Session started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_error(self, error_msg: str, context: str = None):
        """Log an error with optional context."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        error_entry = {
            'timestamp': timestamp,
            'type': 'ERROR',
            'message': error_msg,
            'context': context
        }
        
        self.errors.append(error_entry)
        self._append_to_file(error_entry)
    
    def log_warning(self, warning_msg: str, context: str = None):
        """Log a warning with optional context."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        warning_entry = {
            'timestamp': timestamp,
            'type': 'WARNING',
            'message': warning_msg,
            'context': context
        }
        
        self.warnings.append(warning_entry)
        self._append_to_file(warning_entry)
    
    def _append_to_file(self, entry: Dict[str, Any]):
        """Append entry to log file."""
        with open(self.log_file, 'a') as f:
            f.write(f"[{entry['timestamp']}] {entry['type']}: {entry['message']}\n")
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
        """Write final summary to log file."""
        summary = self.get_summary()
        
        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("SESSION SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Errors: {summary['total_errors']}\n")
            f.write(f"Total Warnings: {summary['total_warnings']}\n")
            f.write(f"Session Duration: {summary['session_duration']}\n")
            f.write("=" * 80 + "\n")

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
