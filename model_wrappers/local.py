import transformers
from base import Base


class Local(Base):
    def _config(self):
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._model)
        self._model = transformers.AutoModelForCausalLM.from_pretrained(self._model)

    def _multinomial_sampling(self, eos_token_id, config: dict):
        top_k = config.pop("top_k")
        temperature = config.pop("temperature")
        min_length = config.pop("min_length")
        max_length = config.pop("max_length")
        logits_warper = transformers.LogitsProcessorList(
            [
                transformers.TopKLogitsWarper(top_k=top_k),
                transformers.TemperatureLogitsWarper(temperature=temperature),
            ]
        )
        logits_processor = transformers.LogitsProcessorList(
            [
                transformers.MinLengthLogitsProcessor(
                    min_length=min_length, eos_token_id=eos_token_id
                )
            ]
        )
        stopping_criteria = transformers.StoppingCriteriaList(
            [transformers.MaxLengthCriteria(max_length=max_length)]
        )
        return {
            "logits_processor": logits_processor,
            "logits_warper": logits_warper,
            "stopping_criteria": stopping_criteria,
        }

    def send_message(
        self,
        input_message: str,
        exclude_input: bool = True,
        method: int = "generate",
        method_kwargs: dict = {},
    ):
        model_input = self._tokenizer(input_message, return_tensors="pt")
        if method == "multinomial_sampling":
            input_config = self._multinomial_sampling(
                eos_token_id=self._model.config.eos_token_id, config=method_kwargs
            )
            model_output = self._model.sample(model_input["input_ids"], **input_config)
        else:
            model_output = self._model.generate(model_input["input_ids"])
        output_token_ids = model_output.detach().flatten()

        if exclude_input:
            input_lengh = len(model_input["input_ids"].detach().flatten())
            output_token_ids = output_token_ids[input_lengh:]
        decoded_answer = self._tokenizer.batch_decode(output_token_ids)
        return decoded_answer
