import logging
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime

class LLMTestLogger:
    def __init__(
        self, 
        log_dir: str = './logs', 
        log_level: int = logging.INFO
    ):
        # Ensure log directory exists
        import os
        os.makedirs(log_dir, exist_ok=True)

        # Create unique log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f'llm_test_{timestamp}.log')

        # Configure logger
        self.logger = logging.getLogger('LLMTestLogger')
        self.logger.setLevel(log_level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_filename, 
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(log_level)

        # Formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )

        # Set formatters
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log_test_start(self, prompt: str, models: List[str]):
        """Log the start of a test run"""
        self.logger.info(f"Starting LLM Test")
        self.logger.info(f"Prompt: {prompt}")
        self.logger.info(f"Models: {', '.join(models)}")

    def log_test_result(self, model: str, metrics: Dict[str, Any]):
        """Log results for a specific model"""
        self.logger.info(f"Test Results for {model}")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value}")

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with optional context"""
        self.logger.error(f"Error occurred: {str(error)}")
        if context:
            self.logger.error(f"Context: {context}")