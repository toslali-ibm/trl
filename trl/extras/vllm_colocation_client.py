from typing import Optional
import torch
import warnings
from vllm import SamplingParams, LLM

from accelerate import PartialState
from profiling import profiling_context

class VLLMColocationClient:
    def __init__(self, accelerator, args, model):
        ## ToDo: get accelerator and other things - to offload if/else from trainer to the clients
        self.args = args
        self.accelerator = accelerator
        self.model = model
        self.vllm_device = None
        self._initialize_colocated_vllm()
        
    def _initialize_colocated_vllm(self):
        device_type = PartialState().default_device.type
        self.vllm_device = f"{device_type}:{self.accelerator.process_index}"

        warnings.warn(
            f"The requested device {self.vllm_device} is also being used for training. For higher throughput "
            "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
            "If this is intentional, you may ignore this warning but should adjust "
            "`vllm_gpu_memory_utilization` accordingly."
        )
           
        self.llm = LLM(
            model=self.model.name_or_path,
            device=self.vllm_device,
            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            dtype=self.args.vllm_dtype,
            enable_prefix_caching=self.args.vllm_enable_prefix_caching,
            max_model_len=self.args.vllm_max_model_len,
            distributed_executor_backend="external_launcher",
        )
        
    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights([(name,weights)])

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> list[list[str]]:
        """
        Generates model completions for the provided prompts.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. `-1` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
                Regular expression to guide the decoding process.

        Returns:
            `list[list[int]]`:
                List of lists of token IDs representing the model-generated completions for each prompt.
        """
        self.sampling_params = SamplingParams(
            max_tokens=self.max_completion_length,
            guided_decoding_regex=guided_decoding_regex,
            n=1, # vLLM on each GPU generates only 1 in vllm_colocation mode
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )
        with profiling_context(self, "vLLM.generate"):
            all_outputs = self.llm.generate(
                prompts, sampling_params=self.sampling_params, use_tqdm=False
            )
        completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
        return completion_ids

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        self.llm.engine.reset_prefix_cache()

