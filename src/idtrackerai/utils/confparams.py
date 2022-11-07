#%%
import logging


class ConfParams:
    dicts = {"base": {"settings_priority": 0}}

    def reorder_dicts(self):
        self.dicts = dict(
            sorted(
                self.dicts.items(),
                key=lambda item: item[1]["settings_priority"],
                reverse=True,
            )
        )
        logging.info(
            f"ConfParams dicts reordered as {list(self.dicts.keys())}"
        )

    @staticmethod
    def pprint_dict(d: dict) -> str:
        test = ""
        pad = min(max([len(key) for key in d.keys()]), 25)
        for key, value in d.items():
            test += f"\n[bold]{key:>{pad}}[/] = {value}"
        return test

    def update_dict(self, data: dict, type: str = "base"):
        data = {key.lower(): value for key, value in data.items()}
        self.dicts[type].update(data)
        logging.info(
            f"Updated '{type}' params with{self.pprint_dict(data)}",
            extra={"markup": True},
        )
        self.reorder_dicts()

    def set_dict(self, data: dict, name: str = "base", priority: int = 1):
        data = {key.lower(): value for key, value in data.items()}
        if "settings_priority" not in data:
            data["settings_priority"] = priority
        self.dicts[name] = data
        logging.info(
            f"ConfParams '{name}' set to{self.pprint_dict(data)}",
            extra={"markup": True},
        )
        self.reorder_dicts()

    def reset_dict(self, type: str = "base"):
        self.dicts[type].clear()
        logging.info(f"{type} params cleared")

    def reset_all(self):
        if any(self.dicts):
            for dict in self.dicts.values():
                dict.clear()
            logging.info("All params cleared")

    def __getattr__(self, name: str):
        lower_name = name.lower()
        for dict in self.dicts.values():
            if lower_name in dict:
                return dict[lower_name]
            print("not in", dict)
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
# %%
