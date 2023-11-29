import logging
import os
import colorlog
import json
import copy
from datetime import datetime


class Logger:
    """
    自定义日志类

    %(levelno)s: 打印日志级别的数值
    %(levelname)s: 打印日志级别名称
    %(pathname)s: 打印当前执行程序的路径，其实就是sys.argv[0]
    %(filename)s: 打印当前执行程序名
    %(funcName)s: 打印日志的当前函数
    %(lineno)d: 打印日志的当前行号
    %(asctime)s: 打印日志的时间
    %(thread)d: 打印线程ID
    %(threadName)s: 打印线程名称
    %(process)d: 打印进程ID
    %(message)s: 打印日志信息
    datefmt:
        %Y  Year with century as a decimal number.
        %m  Month as a decimal number [01,12].
        %d  Day of the month as a decimal number [01,31].
        %H  Hour (24-hour clock) as a decimal number [00,23].
        %M  Minute as a decimal number [00,59].
        %S  Second as a decimal number [00,61].
        %z  Time zone offset from UTC.
        %a  Locale's abbreviated weekday name.
        %A  Locale's full weekday name.
        %b  Locale's abbreviated month name.
        %B  Locale's full month name.
        %c  Locale's appropriate date and time representation.
        %I  Hour (12-hour clock) as a decimal number [01,12].
        %p  Locale's equivalent of either AM or PM.
    """

    _COLORS = {
        "INFO": "green",
        "WARNING": "yellow",
        "DEBUG": "white",
        "ERROR": "red",
    }

    _INFO = logging.INFO
    _DEBUG = logging.DEBUG
    _WARNING = logging.WARNING
    _ERROR = logging.ERROR
    _TEXT_COMPLETION = "text_completion"
    _CHAT_COMPLETION = "chat.completion"

    def __init__(self, filedir, filename, level=logging.DEBUG, is_control=False):
        self.filedir = filedir
        self.filename = filename
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(level)
        self.is_control = is_control
        self.date = datetime.now().date()
        self.handler = None
        self.__init_handler()
        self.cmpl_logger = CompletionLogger(filedir, filename)

    @classmethod
    def build_logger(cls, filepath, filename, level=logging.DEBUG, is_control=False):
        """
        构建日志管理器
        """
        return Logger(filepath, filename, level, is_control)

    def __check_date(self):
        date = datetime.now().date()
        if self.date != date:
            self.date = date
            self.logger.removeHandler(self.handler)
            self.__init_handler()

    def __init_handler(self):
        if self.is_control:
            console_formatter = colorlog.ColoredFormatter(
                fmt="%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
                datefmt='%Y-%m-%d  %H:%M:%S',
                log_colors=self._COLORS
            )
            self.handler = logging.StreamHandler()
            self.handler.setFormatter(console_formatter)
        else:
            file_formatter = logging.Formatter(
                fmt="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S")
            os.makedirs(os.path.join(self.filedir, self.filename), exist_ok=True)
            filepath = os.path.join(self.filedir, self.filename, f"{self.date}.log")
            self.handler = logging.FileHandler(filepath, mode="a", encoding="UTF-8")
            self.handler.setFormatter(file_formatter)
        self.logger.addHandler(self.handler)

    def _log(self, message, level):
        self.__check_date()
        match level:
            case self._INFO:
                self.logger.info(message)
            case self._DEBUG:
                self.logger.debug(message)
            case self._WARNING:
                self.logger.warning(message)
            case self._ERROR:
                self.logger.error(message)
            case self._TEXT_COMPLETION:
                self.cmpl_logger.completion(message, date=self.date, prefix=self._TEXT_COMPLETION)
            case self._CHAT_COMPLETION:
                self.cmpl_logger.completion(message, date=self.date, prefix=self._CHAT_COMPLETION)

    def info(self, message: str):
        self._log(message, self._INFO)

    def debug(self, message: str):
        self._log(message, self._DEBUG)

    def warning(self, message: str):
        self._log(message, self._WARNING)

    def error(self, message: str):
        self._log(message, self._ERROR)

    def text_completion(self, message: str):
        self._log(message, self._TEXT_COMPLETION)

    def chat_completion(self, message: str):
        self._log(message, self._CHAT_COMPLETION)


class CompletionLogger:

    def __init__(self, filedir, filename):
        self.filedir = filedir
        self.filename = filename
        self.cache = []

    def completion(self, message, date, prefix=None):
        filepath = os.path.join(self.filedir, self.filename)
        if prefix:
            filepath = os.path.join(filepath, prefix)
        os.makedirs(filepath, exist_ok=True)
        filepath = os.path.join(filepath, f"{date}.jsonl")
        try:
            for old_filepath, old_message in copy.deepcopy(self.cache):
                with open(old_filepath, mode="a", encoding="utf-8") as f:
                    f.write(old_message+"\n")
                self.cache.remove((old_filepath, old_message))
            with open(filepath, mode="a", encoding="utf-8") as f:
                f.write(message+"\n")
        except OSError or IOError:
            self.cache.append((filepath, message))
