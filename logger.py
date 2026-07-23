"""
Central logging configuration for FeedbackAnalytics.
Import this in any module: from logger import get_logger
"""

import logging
import os
from logging.handlers import RotatingFileHandler




# Directory where log files will be stored
LOG_DIR = "logs"

# Create the log directory if it doesn't already exist
os.makedirs(LOG_DIR, exist_ok=True)

# Full path to the application log file
# Example: logs/feedbackanalytics.log
LOG_FILE = os.path.join(LOG_DIR, "feedbackanalytics.log")

# Read the logging level from the .env file
# If LOG_LEVEL is not defined, default to "WARNING"
# Convert the value to uppercase to match Python logging levels
# Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()

# Define the log message format
_fmt = logging.Formatter(

    # Format of each log entry:
    # Timestamp | Log Level | Logger Name | Log Message
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",

    # Format of the timestamp (YYYY-MM-DD HH:MM:SS)
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create a logger with the given name
def get_logger(name: str) -> logging.Logger:
    """Create and return a configured logger."""
    print("Inside get_logger()")
    # Get or create a logger with the given name
    logger = logging.getLogger(name)

    # Return existing logger if already configured
    if logger.handlers:
        return logger

    # Set the minimum logging level
    logger.setLevel(LOG_LEVEL)

    # Log messages to the console
    console_handler = logging.StreamHandler()
    # Set the formatter for the console handler
    console_handler.setFormatter(_fmt)
    # Add the console handler to the logger
    logger.addHandler(console_handler)

    # Save logs to a rotating file (5 MB, keep 3 backups)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8"
    )
    # Set the formatter for the file handler
    file_handler.setFormatter(_fmt)
    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Prevent duplicate log messages from parent loggers
    logger.propagate = False

    return logger

# Test the logger
if __name__ == "__main__":
    print(f"LOG_LEVEL = {LOG_LEVEL}")
    logger = get_logger(__name__)
    logger.warning("Warning message")