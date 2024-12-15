import argparse
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from livecodebench.runner.giga_base import GigaConfig
from livecodebench.utils.scenarios import Scenario


class Args(BaseModel):
    devices_config: str
    model: str = "GigaChat"
    scenario: Scenario = Scenario.codegeneration
    not_fast: bool = False
    release_version: str = "release_latest"
    cot_code_execution: bool = False
    n: int = 1
    codegen_n: int = 10
    temperature: float = 1.0
    top_p: float = 0.0
    max_tokens: int = 2048
    multiprocess: int = 0
    stop: str = "###"
    continue_existing: bool = False
    continue_existing_with_eval: bool = False
    use_cache: bool = False
    cache_batch_size: int = 100
    debug: bool = False
    evaluate: bool = False
    num_process_evaluate: int = 12
    timeout: int = 6
    custom_output_file: Optional[str] = None
    custom_output_save_name: Optional[str] = None
    output_path: str = ".lcb_output"
    config: Optional[GigaConfig] = None

    @classmethod
    def from_cmd_args(cls, **kwargs):
        args = cls(**kwargs)
        args.stop = args.stop.split(",")

        config = GigaConfig.load_from_config(args.devices_config)
        args.config = config
        args.temperature = config.params.temperature
        args.top_p = config.params.top_p
        args.max_tokens = config.params.max_tokens
        return args


def get_args_from_config_file(config_file) -> Args:
    with open(config_file, encoding="utf-8") as f:
        return Args(**yaml.safe_load(f))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="GigaChat",
        help="Name of the model to use matching `lm_styles.py`",
    )
    # parser.add_argument(
    #     "--local_model_path",
    #     type=str,
    #     default=None,
    #     help="If you have a local model, specify it here in conjunction with --model",
    # )
    # parser.add_argument(
    #     "--trust_remote_code",
    #     action="store_true",
    #     help="trust_remote_code option used in huggingface models",
    # )
    parser.add_argument(
        "--scenario",
        type=Scenario,
        default=Scenario.codegeneration,
        help="Type of scenario to run",
    )
    parser.add_argument(
        "--not_fast",
        action="store_true",
        help="whether to use full set of tests (slower and more memory intensive evaluation)",
    )
    parser.add_argument(
        "--release_version",
        type=str,
        default="release_latest",
        help="whether to use full set of tests (slower and more memory intensive evaluation)",
    )
    parser.add_argument(
        "--cot_code_execution",
        action="store_true",
        help="whether to use CoT in code execution scenario",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of samples to generate"
    )
    parser.add_argument(
        "--codegen_n",
        type=int,
        default=10,
        help="Number of samples for which code generation was run (used to map the code generation file during self-repair)",
    )
    # parser.add_argument(
    #     "--temperature", type=float, default=0.2, help="Temperature for sampling"
    # )
    # parser.add_argument("--top_p", type=float, default=0.95, help="Top p for sampling")
    # parser.add_argument(
    #     "--max_tokens", type=int, default=2000, help="Max tokens for sampling"
    # )
    parser.add_argument(
        "--multiprocess",
        default=0,
        type=int,
        help="Number of processes to use for generation (vllm runs do not use this)",
    )
    parser.add_argument(
        "--stop",
        default="###",
        type=str,
        help="Stop token (use `,` to separate multiple tokens)",
    )
    parser.add_argument("--continue_existing", action="store_true")
    parser.add_argument("--continue_existing_with_eval", action="store_true")
    parser.add_argument(
        "--use_cache", action="store_true", help="Use cache for generation"
    )
    parser.add_argument(
        "--cache_batch_size", type=int, default=100, help="Batch size for caching"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the results")
    parser.add_argument(
        "--num_process_evaluate",
        type=int,
        default=12,
        help="Number of processes to use for evaluation",
    )
    parser.add_argument("--timeout", type=int, default=6, help="Timeout for evaluation")
    # parser.add_argument(
    #     "--openai_timeout", type=int, default=90, help="Timeout for requests to OpenAI"
    # )
    # parser.add_argument(
    #     "--tensor_parallel_size",
    #     type=int,
    #     default=1,
    #     help="Tensor parallel size for vllm",
    # )
    # parser.add_argument(
    #     "--enable_prefix_caching",
    #     action="store_true",
    #     help="Enable prefix caching for vllm",
    # )
    parser.add_argument(
        "--custom_output_file",
        type=str,
        default=None,
        help="Path to the custom output file used in `custom_evaluator.py`",
    )
    parser.add_argument(
        "--custom_output_save_name",
        type=str,
        default=None,
        help="Folder name to save the custom output results (output file folder modified if None)",
    )
    parser.add_argument(
        "--devices_config",
        type=str,
        default=None,
        help="Path to devices config file in yaml format",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=".lcb_output",
        help="Path to script output",
    )
    # parser.add_argument("--dtype", type=str, default="bfloat16", help="Dtype for vllm")

    args = parser.parse_args()

    args.stop = args.stop.split(",")

    config = GigaConfig.load_from_config(args.devices_config)
    args.temperature = config.params.temperature
    args.top_p = config.params.top_p
    args.max_tokens = config.params.max_tokens
    args.output_path = ".lcb_output" if args.output_path is None else args.output_path
    # args.output_path.mkdir(parents=True, exist_ok=True)
    # assert args.tensor_parallel_size != -1

    # if args.multiprocess == -1:
    #     args.multiprocess = os.cpu_count()

    return args


def test():
    args = get_args()
    print(args)


if __name__ == "__main__":
    test()
