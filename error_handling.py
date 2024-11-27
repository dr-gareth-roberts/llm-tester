import logging
import traceback
from typing import Callable, Any

class LLMTestError(Exception):
    """Base exception for LLM testing errors"""
    pass

class ErrorHandler:
    def __init__(self, log_file: str = 'llm_test_errors.log'):
        # Configure logging
        logging.basicConfig(
            filename=log_file, 
            level=logging.ERROR,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('LLMTestErrorHandler')

    def handle_error(self, 
                     error: Exception, 
                     context: Dict[str, Any] = None, 
                     severity: str = 'error'):
        """
        Comprehensive error handling and logging
        
        :param error: Exception that occurred
        :param context: Additional context about the error
        :param severity: Logging severity level
        """
        # Log full traceback
        error_trace = traceback.format_exc()
        
        # Prepare error message
        error_msg = f"Error: {str(error)}\n"
        if context:
            error_msg += f"Context: {context}\n"
        error_msg += f"Traceback:\n{error_trace}"

        # Log based on severity
        log_method = {
            'debug': self.logger.debug,
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical
        }.get(severity.lower(), self.logger.error)

        log_method(error_msg)

    def retry_with_backoff(self, 
                            func: Callable, 
                            max_retries: int = 3, 
                            backoff_factor: float = 2):
        """
        Retry a function with exponential backoff
        
        :param func: Function to retry
        :param max_retries: Maximum number of retries
        :param backoff_factor: Factor to increase wait time
        :return: Result of the function
        """
        import time

        retries = 0
        while retries < max_retries:
            try:
                return func()
            except Exception as e:
                self.handle_error(e, context={'retry': retries})
                wait_time = (backoff_factor ** retries)
                time.sleep(wait_time)
                retries += 1
        
        raise LLMTestError(f"Failed after {max_retries} retries")