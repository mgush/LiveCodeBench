from livecodebench.lm_styles import LMStyle, LanguageModel


def build_runner(args, model: LanguageModel):
    if model.model_style == LMStyle.GigaChat:
        from livecodebench.runner.giga_runner import GigaRunner

        return GigaRunner(args, model)
    if model.model_style == LMStyle.OpenAIChat:
        from livecodebench.runner.oai_runner import OpenAIRunner

        return OpenAIRunner(args, model)
    if model.model_style == LMStyle.OpenAIReason:
        from livecodebench.runner.oai_runner import OpenAIRunner

        return OpenAIRunner(args, model)
    if model.model_style == LMStyle.Gemini:
        from livecodebench.runner.gemini_runner import GeminiRunner

        return GeminiRunner(args, model)
    if model.model_style == LMStyle.Claude3:
        from livecodebench.runner.claude3_runner import Claude3Runner

        return Claude3Runner(args, model)
    if model.model_style == LMStyle.Claude:
        from livecodebench.runner.claude_runner import ClaudeRunner

        return ClaudeRunner(args, model)
    if model.model_style == LMStyle.MistralWeb:
        from livecodebench.runner.mistral_runner import MistralRunner

        return MistralRunner(args, model)
    if model.model_style == LMStyle.CohereCommand:
        from livecodebench.runner.cohere_runner import CohereRunner

        return CohereRunner(args, model)
    if model.model_style == LMStyle.DeepSeekAPI:
        from livecodebench.runner.deepseek_runner import DeepSeekRunner

        return DeepSeekRunner(args, model)
    elif model.model_style in []:
        raise NotImplementedError(
            f"Runner for language model style {model.model_style} not implemented yet"
        )
    else:
        from livecodebench.runner.vllm_runner import VLLMRunner

        return VLLMRunner(args, model)
