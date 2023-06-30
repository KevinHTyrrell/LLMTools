import transformers
from base import Base


class Local(Base):
    def _config(self):
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._model)
        self._model = transformers.AutoModelForCausalLM.from_pretrained(self._model)

    def send_message(
            self,
            input_message: str,
            exclude_input: bool = True
    ):
        model_input = self._tokenizer(input_message, return_tensors='pt')
        model_output = self._model.generate(**model_input)
        output_token_ids = model_output.detach().flatten()

        if exclude_input:
            input_lengh = len(model_input['input_ids'].detach().flatten())
            output_token_ids = output_token_ids[input_lengh:]
        decoded_answer = self._tokenizer.batch_decode(output_token_ids)
        return decoded_answer

