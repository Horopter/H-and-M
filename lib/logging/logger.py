"""
Structured logging system with DEBUG, WARN, ERROR levels and timestamps.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import os


class StructuredLogger:
    """Structured logger with file and console handlers."""
    
    def __init__(
        self,
        name: str,
        log_dir: str = 'logs/',
        log_level: str = 'INFO',
        log_to_file: bool = True,
        log_to_console: bool = True
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically module name)
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARN, ERROR)
            log_to_file: Whether to log to files
            log_to_console: Whether to log to console
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Formatter with timestamp, level, module, and message
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handlers for different log levels
        if log_to_file:
            # Training log (INFO and above)
            training_handler = logging.FileHandler(
                self.log_dir / 'training.log',
                mode='a',
                encoding='utf-8'
            )
            training_handler.setLevel(logging.INFO)
            training_handler.setFormatter(formatter)
            training_handler.addFilter(lambda record: record.levelno >= logging.INFO)
            self.logger.addHandler(training_handler)
            
            # Error log (ERROR only)
            error_handler = logging.FileHandler(
                self.log_dir / 'errors.log',
                mode='a',
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            self.logger.addHandler(error_handler)
            
            # Debug log (all levels)
            debug_handler = logging.FileHandler(
                self.log_dir / 'debug.log',
                mode='a',
                encoding='utf-8'
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(formatter)
            self.logger.addHandler(debug_handler)
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def debug(self, message: str, *args, **kwargs):
        """Log DEBUG message."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log INFO message."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log WARNING message."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log ERROR message."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log CRITICAL message."""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, *args, **kwargs)


# Global logger instances
_loggers: dict = {}


def get_logger(name: str, **kwargs) -> StructuredLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
        **kwargs: Additional arguments for StructuredLogger
        
    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, **kwargs)
    return _loggers[name]

