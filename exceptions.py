"""
Custom exceptions for FeedbackIQ.
Import in any module: from exceptions import ModelNotLoadedError
"""

class FeedBackError(Exception):
    """Base exception for all Feedback exceptions."""
    # Initialize the exception with a message
    def __init__(self,message:str):
        super().__init__(message)
        self.message = message
    
    # Return a string representation of the exception
    def __str__(self):
        return f"[{self.__class__.__name__}] {self.message}"
    
# Exception for data related errors
class DataError(FeedBackError):
    """Exception for data related errors."""

class DataNotFoundError(DataError):
    """Exception for when processed data file is not found."""
    def __init__(self, path=""):
        if path:
            message = f"Processed data not found at '{path}'. Run: python scripts/preprocess.py"
        else:
            message = "Processed data file not found. Run: python scripts/preprocess.py"

        super().__init__(message)
        self.path = path
