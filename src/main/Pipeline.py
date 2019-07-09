import json
import importlib

from src.utility.Config import Config
from src.utility.Utility import Utility
import re

class Pipeline:

    def __init__(self, config_path, args):
        self.modules = []

        with open(Utility.resolve_path(config_path), "r") as f:
            json_text = f.read()
            # Remove comments
            json_text = re.sub(r'^//.*\n?', '', json_text, flags=re.MULTILINE)
            # Replace arguments
            for i, arg in enumerate(args):
                json_text = json_text.replace("<args:" + str(i) + ">", arg)

            if "<args:" in json_text:
                raise Exception("Too less arguments given")

            config = json.loads(json_text)

        global_config = config["global"]

        for module_config in config["modules"]:
            model_type = module_config["name"].split(".")[0]
            base_config = global_config[model_type] if model_type in global_config else {}
            config = module_config["config"]
            Utility.merge_dicts(base_config, config)

            with Utility.BlockStopWatch("Initializing module " + module_config["name"]):
                module_class = getattr(importlib.import_module("src." + module_config["name"]), module_config["name"].split(".")[-1])
                self.modules.append(module_class(Config(config)))

    def run(self):
        for module in self.modules:

            with Utility.BlockStopWatch("Running module " + module.__class__.__name__):
                module.run()