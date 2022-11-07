import logging


class ConfParams:
    dicts = {}

    def reorder_dicts(self):
        old_order = list(self.dicts.keys())
        self.dicts = dict(
            sorted(
                self.dicts.items(),
                key=lambda item: item[1]["settings_priority"],
                reverse=True,
            )
        )
        new_order = list(self.dicts.keys())
        if new_order != old_order:
            logging.info(f"ConfParams reordered as {new_order}")

    @staticmethod
    def pprint_dict(d: dict) -> str:
        text = ""
        pad = min(max([len(key) for key in d.keys()]), 25)
        for key, value in d.items():
            text += f"\n[bold]{key:>{pad}}[/] = {value}"
        return text

    def update_dict(self, data: dict, type: str = "base"):
        data = {key.lower(): value for key, value in data.items()}
        self.dicts[type].update(data)
        logging.info(
            f"'{type}' updated with:{self.pprint_dict(data)}",
            extra={"markup": True},
        )
        self.reorder_dicts()

    def set_dict(
        self, data: dict, name: str = "base", priority: int = 1, verbose=True
    ):
        data = {key.lower(): value for key, value in data.items()}
        if "settings_priority" not in data:
            data["settings_priority"] = priority
        self.dicts[name] = data
        if verbose:
            logging.info(
                f"'{name}' parameters set to:{self.pprint_dict(data)}",
                extra={"markup": True},
            )
        else:
            logging.info(f"'{name}' parameters set")
        self.reorder_dicts()

    def reset_dict(self, type: str = "base"):
        settings_priority = self.dicts[type]["settings_priority"]
        self.dicts[type].clear()
        self.dicts[type]["settings_priority"] = settings_priority
        logging.info(f"{type} params cleared")

    def reset_all(self):
        if any(self.dicts):
            for dict in self.dicts.values():
                settings_priority = dict["settings_priority"]
                dict.clear()
                dict["settings_priority"] = settings_priority
            logging.info("All params cleared")

    def __getattr__(self, name: str):
        lower_name = name.lower()
        for dict in self.dicts.values():
            if lower_name in dict:
                return dict[lower_name]
        raise AttributeError(f"ConfParams object has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        lower_name = name.lower()
        for dict in self.dicts.values():
            if lower_name in dict:
                logging.error(
                    f"Can't set '{name}' = {value}. "
                    "To change the parameters values, use the class methods"
                )
        object.__setattr__(self, name, value)


conf = ConfParams()
