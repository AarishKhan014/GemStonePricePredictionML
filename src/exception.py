import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file = exc_tb.tb_frame.f_code.co_filename
    line = exc_tb.tb_lineno
    return f"Error in [{file}] at line [{line}]: {error}"

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message