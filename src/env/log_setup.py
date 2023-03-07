import os
import logging

FORMATTER = logging.Formatter('%(message)s')


def logger_setup(log, alias, log_dir="logs"):
    # check log dir exists
    log_path = os.path.join(os.getcwd(), log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_path)

    # Episode log
    #full_path = os.path.join(log_path, alias + "_episode" + '.log')
    #fh_episode = logging.FileHandler(full_path, mode='w')
    #fh_episode.setLevel(logging.WARNING)
    #fh_episode.setFormatter(FORMATTER)
    #log.addHandler(fh_episode)  # set the new handler

    # Step log
    full_path = os.path.join(log_path, alias + "_step" + '.log')
    fh_step = logging.FileHandler(full_path, mode='w')
    fh_step.setLevel(logging.WARNING)
    fh_step.setFormatter(FORMATTER)
    log.addHandler(fh_step)  # set the new handler

    return log
