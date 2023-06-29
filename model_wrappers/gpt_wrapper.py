import openai
import os
from typing import Union
from base_wrapper import BaseWrapper
openai.api_key = os.getenv('OPENAI_API_KEY')


class GPTWrapper(BaseWrapper):
    def _config(self):
        self._message_list.append({'role': 'system', 'content': 'You are a helpful assistant.'})

    def _get_input_formatted(self, input_str: str, use_previous_results: bool = False):
        input_formatted = {'role': 'user', 'content': input_str}
        self._message_list.append(input_formatted)
        if not use_previous_results:
            return [input_formatted]
        else:
            return self._message_list
    def send_message(
            self,
            input_str: str,
            temperature: Union[int, float] = 0,
            use_previous_results: bool = False,
            return_full_response: bool = False
    ):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        message_input = self._get_input_formatted(input_str=input_str, use_previous_results=use_previous_results)
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=message_input,
            temperature=temperature,
        )
        response_str = response['choices'][0]['message']['content']
        self._message_list.append({'role': 'assistant', 'content': response_str})
        return response if return_full_response else response_str

