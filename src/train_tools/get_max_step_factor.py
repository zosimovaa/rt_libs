def get_max_step_factor(config):
    step_factor = 1
    if isinstance(config, dict):
        keys = list(config.keys())

        for key in keys:
            if isinstance(config[key], dict):
                step_factor = max(step_factor, get_max_step_factor(config[key]))

            if isinstance(config[key], list):
                for list_item in config[key]:
                    step_factor = max(step_factor, get_max_step_factor(list_item))

            if key == "step_factor":
                step_factor = max(step_factor, config[key])

    return step_factor