from copy import deepcopy
import json
import logging
import os
from typing import Optional

from livecodebench.evaluation import extract_instance_results
from livecodebench.lm_styles import LanguageModelStore
from livecodebench.runner.parser import (Args, get_args,
                                         get_args_from_config_file)
from livecodebench.runner.runner_utils import build_runner
from livecodebench.runner.scenario_router import (
    build_prompt_benchmark, combine_results, get_metrics,
    sort_and_extract_save_results)
from livecodebench.utils.path_utils import get_output_path
from livecodebench.utils.scenarios import Scenario

LOGGING_FMT = "[%(asctime)s][%(levelname)-8s][%(name)-24s] %(message)s"
logging.basicConfig(filename="runner.log", filemode="w", level=logging.DEBUG, format=LOGGING_FMT, force=True)
logger = logging.getLogger(__name__)

def main(
    scenario: str,
    devices_config: str = ".giga_config.yml",
    release_version: str = "release_latest",
    cot_code_execution: bool = False,
    evaluate: bool = False,
    debug: bool = False,
    output_path: str = ".lcb_output",
    codegen_n: int = 10,
    continue_existing: bool = True,
    continue_existing_with_eval: bool = True,
):
    # args = get_args_from_config_file(".config.yml")
    # args = get_args()
    args = Args.from_cmd_args(
        scenario=scenario,
        devices_config=devices_config,
        release_version=release_version,
        evaluate=evaluate,
        debug=debug,
        output_path=output_path,
        codegen_n=codegen_n,
        continue_existing=continue_existing,
        continue_existing_with_eval=continue_existing_with_eval,
    )
    # print(args)
    model = deepcopy(LanguageModelStore[args.model])
    # model = 
    model.model_repr = f"{model.model_repr}_{args.config.model}"
    # print(model.model_repr)
    benchmark, format_prompt = build_prompt_benchmark(args)
    if args.debug:
        logger.info("Running with %s instances in debug mode", len(benchmark))
        # print(f"Running with {len(benchmark)} instances in debug mode")
        benchmark = benchmark[:5]

    output_path = get_output_path(model.model_repr, args)
    eval_file = output_path.replace(".json", "_eval.json")
    eval_all_file = output_path.replace(".json", "_eval_all.json")

    if args.continue_existing or args.continue_existing_with_eval:
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                old_save_results = json.load(f)
        elif os.path.exists(eval_all_file):
            with open(eval_all_file, "r") as f:
                old_save_results = json.load(f)
        else:
            # print(
            #     f"File {output_path} does not exist in --continue_existing, starting from scratch"
            # )
            logger.info("File %s does not exist in --continue_existing, starting from scratch", output_path)
            old_save_results = []

        old_save_results = [
            instance
            for instance in old_save_results
            if instance["output_list"] and [x for x in instance["output_list"] if x]
        ]
        old_save_results_question_ids = [
            instance["question_id"] for instance in old_save_results
        ]
        remaining_benchmark = [
            instance
            for instance in benchmark
            if instance.question_id not in old_save_results_question_ids
        ]
        # print(
        #     f"Found {len(old_save_results)} existing generations, continuing with {len(remaining_benchmark)} remaining"
        # )
        logger.info("Found %s existing generations, continuing with %s remaining", len(old_save_results), len(remaining_benchmark))
    else:
        old_save_results = []
        remaining_benchmark = benchmark

    if len(remaining_benchmark) > 0:
        runner = build_runner(args, model)
        results: list[list[str]] = runner.run_main(remaining_benchmark, format_prompt)
    else:
        results = []

    combined_results = combine_results(
        args.scenario, results, model, args.cot_code_execution
    )

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            remaining_benchmark, combined_results
        )
    ]

    if args.continue_existing or args.continue_existing_with_eval:
        save_results += old_save_results

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    if args.evaluate:
        if args.continue_existing_with_eval and os.path.exists(eval_all_file):
            with open(eval_all_file) as fp:
                old_eval_all_results = json.load(fp)

            if os.path.exists(eval_file):
                with open(eval_file) as fp:
                    old_eval_results = json.load(fp)
            else:
                old_eval_results = None

            old_eval_results_question_ids = [
                instance["question_id"] for instance in old_eval_all_results
            ]
            remaining_indices = [
                idx
                for idx in range(len(benchmark))
                if benchmark[idx].question_id not in old_eval_results_question_ids
            ]
            benchmark = [benchmark[idx] for idx in remaining_indices]
            combined_results = [combined_results[idx] for idx in remaining_indices]

            old_eval_size = len(old_eval_results_question_ids)
            new_eval_size = len(benchmark)

            if new_eval_size == 0:
                return

            logger.info("Found %s, running evals for %s problems", old_eval_size, new_eval_size)
            # print(f"Found {old_eval_size}, running evals for {new_eval_size} problems")

            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])

            if old_eval_results:
                for key in metrics[0]:
                    if key in old_eval_results[0]:
                        if key != "detail":
                            metrics[0][key] = (
                                old_eval_size * old_eval_results[0][key]
                                + new_eval_size * metrics[0][key]
                            )
                            metrics[0][key] /= old_eval_size + new_eval_size

                for key in metrics[0]["detail"]:
                    if key in old_eval_results[0]["detail"]:
                        metrics[0]["detail"][key] = {
                            **metrics[0]["detail"][key],
                            **old_eval_results[0]["detail"][key],
                        }
                metrics[1] = {**metrics[1], **old_eval_results[1]}
            else:
                # print("Old eval file not present, cannot update eval file")
                logger.warning("Old eval file not present, cannot update eval file")
                metrics = {}

        else:
            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])
            old_eval_all_results = []
            old_eval_results = []

        if args.scenario == Scenario.codegeneration:
            if metrics:
                metadatas = metrics[2]
            else:
                metadatas = [[] for _ in benchmark]
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list, metadata=meta
                )
                for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                    benchmark, combined_results, graded, metadatas
                )
            ]
            if metrics and old_eval_results:
                old_eval_results
                metrics[2] = old_eval_results[2] + metrics[2]
        elif args.scenario == Scenario.selfrepair:
            metadatas = metrics[2]
            with open(
                f"{args.output_path}/{model.model_repr}/{Scenario.codegeneration}_{args.codegen_n}_{args.temperature}_eval_all.json"
            ) as f:
                code_gen_evals = json.load(f)
            original_code_lists = [
                code_gen_eval["code_list"] for code_gen_eval in code_gen_evals
            ]

            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list,
                    extracted_list,
                    graded_list,
                    metadata=meta,
                    original_code_list=original_code_list,
                )
                for instance, (
                    outputs_list,
                    extracted_list,
                ), graded_list, meta, original_code_list in zip(
                    benchmark, combined_results, graded, metadatas, original_code_lists
                )
            ]

        else:
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list
                )
                for instance, (outputs_list, extracted_list), graded_list in zip(
                    benchmark, combined_results, graded
                )
            ]

        save_eval_results = old_eval_all_results + save_eval_results

        with open(eval_file, "w") as f:
            json.dump(metrics, f, indent=4)

        with open(eval_all_file, "w") as f:
            json.dump(save_eval_results, f, indent=4)


if __name__ == "__main__":
    main()
