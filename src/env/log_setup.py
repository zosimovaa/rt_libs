import os
import logging

FORMATTER = logging.Formatter('%(message)s')


def logger_factory(logger_name, alias, log_dir="logs"):

    log_path = os.path.join(os.getcwd(), log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_path)

    logger = logging.getLogger("env." + logger_name)

    full_path = os.path.join(log_path, alias + "_" + logger_name + '.log')
    file_handler = logging.FileHandler(full_path, mode='w')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(FORMATTER)
    logger.addHandler(file_handler)  # set the new handler

    return logger


if __name__ == "__main__":
    logger_episode = logger_factory("episode", "TEST_ALIAS")
    logger_step = logger_factory("step", "TEST_ALIAS")

    print(logger_episode.handlers)
    logger_episode.info("INFO")
    logger_episode.warning("WARNING")

    logger_step.info("INFO")
    logger_step.warning("WARNING")
