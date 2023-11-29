from serve.utils.logger import Logger

logger = Logger.build_logger("./test_logger", "logger")
logger.info("hello logger!")
logger.text_completion({
    "hello": "world"
})