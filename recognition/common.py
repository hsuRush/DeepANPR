from argparse import ArgumentTypeError
import logging
import os

import numpy as np


# Commandline argument parsing
###
def probility_float(arg):
    try:
        value = float(arg)
    except ValueError:
        raise ArgumentTypeError('The argument "{}" is not an {}.'.format(
            arg, float.__name__))

    if 1 > value > 0:
        return value
    else:
        raise ArgumentTypeError('Found {} where an {} greater than 0 and lesser than 1 was '
            'required'.format(arg, float.__name__, x))

def positive_int(arg):
    return number_greater_x(arg, int, 0)


def nonnegative_int(arg):
    return number_greater_x(arg, int, -1)


def positive_float(arg):
    return number_greater_x(arg, float, 0)

def number_greater_x(arg, type_, x):
    try:
        value = type_(arg)
    except ValueError:
        raise ArgumentTypeError('The argument "{}" is not an {}.'.format(
            arg, type_.__name__))

    if value > x:
        return value
    else:
        raise ArgumentTypeError('Found {} where an {} greater than {} was '
            'required'.format(arg, type_.__name__, x))

def float_or_string(arg):
    """Tries to convert the string to float, otherwise returns the string."""
    try:
        return float(arg)
    except (ValueError, TypeError):
        return arg

def get_logging_dict(name):
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'stderr': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'common.ColorStreamHandler',
                'stream': 'ext://sys.stderr',
            },
            'logfile': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': name + '.log',
                'mode': 'a',
            }
        },
        'loggers': {
            '': {
                'handlers': ['stderr', 'logfile'],
                'level': 'DEBUG',
                'propagate': True
            },

            # extra ones to shut up.
            'keras': {
                'handlers': ['stderr', 'logfile'],
                'level': 'INFO',
            },
        }
    }
class _AnsiColorStreamHandler(logging.StreamHandler):
    DEFAULT = '\x1b[0m'
    RED     = '\x1b[31m'
    GREEN   = '\x1b[32m'
    YELLOW  = '\x1b[33m'
    CYAN    = '\x1b[36m'

    CRITICAL = RED
    ERROR    = RED
    WARNING  = YELLOW
    INFO     = DEFAULT  # GREEN
    DEBUG    = CYAN

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:  return cls.CRITICAL
        elif level >= logging.ERROR:   return cls.ERROR
        elif level >= logging.WARNING: return cls.WARNING
        elif level >= logging.INFO:    return cls.INFO
        elif level >= logging.DEBUG:   return cls.DEBUG
        else:                          return cls.DEFAULT

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)

    def format(self, record):
        text = logging.StreamHandler.format(self, record)
        color = self._get_color(record.levelno)
        return (color + text + self.DEFAULT) if self.is_tty() else text

    def is_tty(self):
        isatty = getattr(self.stream, 'isatty', None)
        return isatty and isatty()


class _WinColorStreamHandler(logging.StreamHandler):
    # wincon.h
    FOREGROUND_BLACK     = 0x0000
    FOREGROUND_BLUE      = 0x0001
    FOREGROUND_GREEN     = 0x0002
    FOREGROUND_CYAN      = 0x0003
    FOREGROUND_RED       = 0x0004
    FOREGROUND_MAGENTA   = 0x0005
    FOREGROUND_YELLOW    = 0x0006
    FOREGROUND_GREY      = 0x0007
    FOREGROUND_INTENSITY = 0x0008 # foreground color is intensified.
    FOREGROUND_WHITE     = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED

    BACKGROUND_BLACK     = 0x0000
    BACKGROUND_BLUE      = 0x0010
    BACKGROUND_GREEN     = 0x0020
    BACKGROUND_CYAN      = 0x0030
    BACKGROUND_RED       = 0x0040
    BACKGROUND_MAGENTA   = 0x0050
    BACKGROUND_YELLOW    = 0x0060
    BACKGROUND_GREY      = 0x0070
    BACKGROUND_INTENSITY = 0x0080 # background color is intensified.

    DEFAULT  = FOREGROUND_WHITE
    CRITICAL = BACKGROUND_YELLOW | FOREGROUND_RED | FOREGROUND_INTENSITY | BACKGROUND_INTENSITY
    ERROR    = FOREGROUND_RED | FOREGROUND_INTENSITY
    WARNING  = FOREGROUND_YELLOW | FOREGROUND_INTENSITY
    INFO     = FOREGROUND_GREEN
    DEBUG    = FOREGROUND_CYAN

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:  return cls.CRITICAL
        elif level >= logging.ERROR:   return cls.ERROR
        elif level >= logging.WARNING: return cls.WARNING
        elif level >= logging.INFO:    return cls.INFO
        elif level >= logging.DEBUG:   return cls.DEBUG
        else:                          return cls.DEFAULT

    def _set_color(self, code):
        import ctypes
        ctypes.windll.kernel32.SetConsoleTextAttribute(self._outhdl, code)

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)
        # get file handle for the stream
        import ctypes, ctypes.util
        # for some reason find_msvcrt() sometimes doesn't find msvcrt.dll on my system?
        crtname = ctypes.util.find_msvcrt()
        if not crtname:
            crtname = ctypes.util.find_library("msvcrt")
        crtlib = ctypes.cdll.LoadLibrary(crtname)
        self._outhdl = crtlib._get_osfhandle(self.stream.fileno())

    def emit(self, record):
        color = self._get_color(record.levelno)
        self._set_color(color)
        logging.StreamHandler.emit(self, record)
        self._set_color(self.FOREGROUND_WHITE)

# select ColorStreamHandler based on platform
import platform
if platform.system() == 'Windows':
    ColorStreamHandler = _WinColorStreamHandler
else:
    ColorStreamHandler = _AnsiColorStreamHandler
    