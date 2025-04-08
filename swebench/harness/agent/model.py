from swebench.harness.agent.prompt import Prompt
from swebench.harness.agent.retriever import Retriever
from swebench.harness.constants.swing_constants import SwingbenchInstance

from openai import OpenAI


class ModelInfo:
    def __init__(self, name: str, base_url: str = None, api_key: str = None, system_msg_tpl: str = None, user_prompt_tpl: str = None):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.system_msg_tpl = None
        self.user_prompt_tpl = None
        if system_msg_tpl is not None:
            self.system_msg_tpl = system_msg_tpl
        if user_prompt_tpl is not None:
            self.user_prompt_tpl = user_prompt_tpl

class AgentProxy:
    def __init__(self, model_info: ModelInfo):
        """
        Initialize agent proxy.

        Args:
            name (str): agent type
        """
        self.model_info = model_info
        self.score = 0

    def _call_api(self, prompt: Prompt):
        """
        Route the prompt to different API.

        Args:
            prompt (str): your prompt
        """
        response = None
        # TODO(haoran): offline server
        response = self._call_openai(prompt)
        # if self.model_info.name in OPENAI_LIST:
        #     response = self._call_openai(prompt)
        # else:
        #     # TODO(wdxu): offline server
        #     response = self._call_offline()
        return response

    def _call_openai(self, prompt: Prompt):
        """
        Openai interface.

        Args:
            base_url (str): the base url of the openai server e.g. http://localhost:8000/v1
            prompt (str): your prompt
        """
        # For local inference, we don't need an API key
        client = OpenAI(
            api_key=self.model_info.api_key,
            base_url=self.model_info.base_url,
        )
        response = client.chat.completions.create(
            model=self.model_info.name,
            messages=[
                {"role": "system", "content": prompt.system_message},
                {"role": "user", "content": prompt.user_prompt},
            ],
        )

        return response

    def _call_offline(self):
        raise NotImplementedError("Offline server is not implemented yet.")

    def generate_patch(self, data: SwingbenchInstance, system_msg: str, user_prompt_tpl: str, retriever: Retriever = None):
        """
        Patch generater.

        Args:
            data (SwingbenchInstance): a piece of data from dataset
        """
        issue = data.problem_statement + "\n" + data.hints_text
        code_snippset = None
        if retriever is not None:
            # get code_snippset from retriever
            code_snippset = retriever.retrieve(data)
        prompt = Prompt()
        prompt.system_message = system_msg
        prompt.user_prompt = user_prompt_tpl.format(issue=issue, code_snippset=code_snippset)

        return self._call_api(prompt)

    def generate_test(self, data: SwingbenchInstance, system_msg: str, user_prompt_tpl: str, retriever: Retriever = None):
        """
        Test generater.

        Args:
            data (SwingbenchInstance): a piece of data from dataset
        """
        issue = data.problem_statement + "\n" + data.hints_text
        patch = data.patch
        sample = data.test_patch
        code_snippset = None
        if retriever is not None:
            # get code_snippset from retriever
            code_snippset = retriever.retrieve(data)
        prompt = Prompt()
        prompt.system_message = system_msg
        prompt.user_prompt = user_prompt_tpl.format(issue=issue, code_snippset=code_snippset, patch=patch, sample=sample)

        return self._call_api(prompt)


if __name__ == "__main__":
    import swing_utils
    from swebench.harness.agent.retriever import BM25DiskRetriever
    from swebench.harness.agent.prompt import GENERATE_PATCH_SYSTEM_MESSAGE, GENERATE_PATCH_TEMPLATE

    retriever = BM25DiskRetriever(index_dir="/mnt/Data/wdxu/github/Swing-Bench/tmpdata/indexes")
    dataset_jsonl_path = '/mnt/Data/wdxu/github/Swing-Bench/tmpdata/dataset.json'
    dataset = swing_utils.load_swingbench_dataset(dataset_jsonl_path)

    model_info = ModelInfo(name="/home/mnt/wdxu/models/DeepSeek-R1-Distill-Qwen-7B", base_url="http://localhost:8000/v1", api_key="no-api-key")
    agent_proxy = AgentProxy(model_info)
    
    for instance in dataset:
        response = agent_proxy.generate_patch(instance, GENERATE_PATCH_SYSTEM_MESSAGE, GENERATE_PATCH_TEMPLATE, retriever)
        print(response)
        break