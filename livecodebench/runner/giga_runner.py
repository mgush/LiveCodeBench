import logging
import os
from time import sleep

from .giga_base import GigaChat, GigaConfig

from livecodebench.lm_styles import LMStyle
from livecodebench.runner.base_runner import BaseRunner

logger = logging.getLogger(__name__)

class GigaRunner(BaseRunner):
    client = GigaChat()

    def __init__(self, args, model):
        super().__init__(args, model)
        self.client_kwargs: dict[str | str] = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
        }
        # assert args.devices_config is not None, "Provide devices config"
        # if GigaRunner.client.config is None:
        #     config = GigaConfig.load_from_config(args.devices_config)
        #     GigaRunner.client.update_config(config=config)

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        assert isinstance(prompt, list)
        results = []
        try:
            answers = GigaRunner.client.create_chat_completion(
                prompt=prompt, **self.client_kwargs
            )
            results.extend(answers)

        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
        return results
