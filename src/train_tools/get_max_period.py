def get_max_period(config):
    period = 1
    if isinstance(config, dict):
        keys = list(config.keys())

        for key in keys:
            if isinstance(config[key], dict):
                period = max(period, get_max_period(config[key]))

            if isinstance(config[key], list):
                for list_item in config[key]:
                    period = max(period, get_max_period(list_item))

            if key == "period":
                period = max(period, config[key])

    return period
