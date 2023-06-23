import logging
from typing import Any


class ConfParams:
    parameters: dict = {}

    @staticmethod
    def pprint_dict(d: dict) -> str:
        text = ""
        pad = min(max(map(len, d.keys())), 25)
        for key, value in d.items():
            text += f"\n[bold]{key:>{pad}}[/] = {repr(value)}"
        return text

    def set_dict(self, data: dict[str, Any], verbose=False):
        logging.info("Setting %d keys in ConfParams", len(data))
        data = {key.lower(): value for key, value in data.items()}
        if verbose:
            logging.info(self.pprint_dict(data), extra={"markup": True})
        self.parameters = data

    def __getattr__(self, name: str):
        lower_name = name.lower()

        if lower_name not in self.parameters:
            raise AttributeError(f"ConfParams object has no attribute '{name}'")
        return self.parameters[lower_name]

    def __setattr__(self, name: str, value):
        lower_name = name.lower()
        if lower_name in self.parameters:
            logging.error(
                f"Can't set '{name}' = {value}. "
                "To change the parameters values, use the class methods"
            )
        object.__setattr__(self, name, value)


conf = ConfParams()
