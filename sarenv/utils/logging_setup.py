# sarenv/utils/logging_setup.py
import inspect
import logging
from pathlib import Path
import colorama # Ensure colorama is in requirements.txt

LOGGER_NAME = "sarenv" # Package-level logger name

# Store initialized state to prevent multiple handler attachments
_logger_initialized = False

def init_logger(level=None, log_format="%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"):
    global _logger_initialized
    if level is None:
        import os
        _env = os.environ.get("SARENV_LOG_LEVEL", "DEBUG").upper()
        level = getattr(logging, _env, logging.DEBUG)
    if _logger_initialized:
        # logging.getLogger(LOGGER_NAME).debug("Logger already initialized.")
        return logging.getLogger(LOGGER_NAME)

    try:
        colorama.init(autoreset=True) # autoreset=True simplifies usage
    except Exception as e:
        # Fallback if colorama init fails (e.g. in certain restricted environments)
        print(f"Colorama initialization failed: {e}. Proceeding without colors.")


    LOG_COLORS = {
        "TRACE": colorama.Fore.MAGENTA if 'colorama' in globals() else '', # Check if colorama loaded
        "DEBUG": colorama.Fore.BLUE if 'colorama' in globals() else '',
        "INFO": colorama.Fore.GREEN if 'colorama' in globals() else '',
        "WARNING": colorama.Fore.YELLOW if 'colorama' in globals() else '',
        "ERROR": colorama.Fore.RED if 'colorama' in globals() else '',
        "CRITICAL": colorama.Fore.RED + colorama.Style.BRIGHT if 'colorama' in globals() else '',
    }
    RESET_ALL = colorama.Style.RESET_ALL if 'colorama' in globals() else ''


    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            log_level_name = record.levelname
            color = LOG_COLORS.get(log_level_name, "")
            
            # Simplification: filename and lineno are already part of the default log_format string
            # If you need to get them from inspect for a specific reason, ensure it's robust.
            # For now, relying on standard record attributes.
            # frame_info = inspect.stack()[-1] # This can be slow and sometimes points to logging internals
            # filename = Path(frame_info[1])
            # record.filename = filename.name # This overrides the actual record.filename

            original_levelname = record.levelname
            record.levelname = f"{color}{original_levelname}{RESET_ALL}"
            formatted_message = super().format(record)
            record.levelname = original_levelname # Reset for other handlers if any
            return formatted_message

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    # Prevent adding multiple handlers if this function is somehow called again by mistake
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = ColoredFormatter(log_format)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        # Add a TRACE level if needed (lower than DEBUG)
        logging.addLevelName(5, "TRACE")
        def trace(self, message, *args, **kws):
            if self.isEnabledFor(5):
                self._log(5, message, args, **kws)
        logging.Logger.trace = trace

    _logger_initialized = True
    return logger


def get_logger(name=LOGGER_NAME):
    """
    Retrieves the logger instance. Initializes it if not already done.
    """
    logger = logging.getLogger(name)
    if not _logger_initialized or not logger.handlers : # Also check handlers for re-entry robustness
        # print(f"Logger '{name}' not fully initialized or no handlers, re-initializing.")
        init_logger(level=logger.level or logging.DEBUG) # Use current level if already set
    return logger


# Initialize logger when module is loaded if not already done by an explicit call
if not _logger_initialized:
    init_logger()

# Example usage (can be removed from here and put in tests/examples)
# if __name__ == "__main__":
#     log = get_logger()
#     log.trace("This is a trace message") # Example, requires TRACE level
#     log.debug("This is a debug message")
#     log.info("This is an info message")
#     log.warning("This is a warning message")
#     log.error("This is an error message")
#     log.critical("This is a critical message")