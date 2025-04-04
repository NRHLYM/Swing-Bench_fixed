import os

from abc import ABC, abstractmethod
from pathlib import Path
from openai import OpenAI
from swebench.harness.constants.swing_constants import(
    AgentState,
    SwingbenchInstance
)
from swebench.inference.make_datasets.swing_search_index import search_instance

OPENAI_LIST = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4.5-preview",
               "/home/mnt/wdxu/models/DeepSeek-R1-Distill-Qwen-7B",
               "/home/mnt/wdxu/models/Qwen2.5-Coder-14B-Instruct",
               "/home/mnt/wdxu/models/Qwen2.5-Coder-32B-Instruct",
               "/app/wdxu/models/Qwen2.5-Coder-32B",
               "/app/wdxu/models/DeepSeek-R1-Distill-Qwen-32B"]

MODEL_LIMITS = {
    # "claude-instant-1": 100_000,
    # "claude-2": 100_000,
    # "claude-3-opus-20240229": 200_000,
    # "claude-3-sonnet-20240229": 200_000,
    # "claude-3-haiku-20240307": 200_000,
    "gpt-3.5-turbo": 16_385,
    "gpt-4": 8_192,
    "gpt-4o": 128_000,
    "gpt-4.5-preview": 128_000,
}

# change to a more efficient template
GENERATE_PATCH_SYSTEM_MESSAGE = "You are an AI Senior Full-Stack Engineer specialized in GitHub issue triage and bug fixing." \
                                "You should only generate the patch code, without any other text. Provide the .patch format code that could be `git apply` to the original code."
GENERATE_PATCH_TEMPLATE = "You are required to write a patch for the specified issue and its corresponding code section." \
                          "The issue details: {issue} " \
                          "The code snippet: {code_snippset} "
# change to a more efficient template
GENERATE_TEST_SYSTEM_MESSAGE = "You are an AI Test Automation Engineer specializing in generating unit tests for code patches." \
                                "You should only generate the test code, without any other text. Provide the .patch format code that could be `git apply` to original code or create a new test file."
GENERATE_TEST_TEMPLATE = "You are required to develop unit tests for the specified patch, which was created to resolve this issue." \
                          "The issue details: {issue} " \
                          "The code snippet: {code_snippset} " \
                          "The patch: {patch} " \
                          "The test case sample: {sample} "


class Retriever:
    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Retriever is not implemented yet.")

    @abstractmethod
    def retrieve(self, instance: SwingbenchInstance):
        raise NotImplementedError("Retrieve is not implemented yet.")


class BM25DiskRetriever(Retriever):
    def __init__(self, index_dir: str, document_encoding_style: str = "file_name_and_contents"):
        self.index_dir = Path(index_dir)
        self.document_encoding_style = document_encoding_style

    def retrieve(self, instance: SwingbenchInstance):
        results = search_instance(
            instance,
            self.index_dir,
            self.document_encoding_style,
            k=1
        )
        # TODO(wdxu): need some reduce strategies
        
        return results


class Verifier:
    def __init__(self):
        pass

    def verify_patch(self, data: SwingbenchInstance, patch: str, test: str = None):
        """
        Patch verifier.

        Args:
            data (SwingbenchInstance): a piece of data from dataset
            patch: patch string? temporary patch file?
            test: test string? temporary test file?
        
        """
        pass

    def verify_test(self, data: SwingbenchInstance, test: str):
        """
        Test verifier.

        Args:
            data (SwingbenchInstance): a piece of data from dataset
            test: test string? temporary test file?
        """
        if os.path.exists(test):
            # test is a file
            pass
        else:
            # test is a string
            pass


class ModelInfo:
    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url


class AgentProxy:
    def __init__(self, model_info: ModelInfo):
        """
        Initialize agent proxy.

        Args:
            name (str): agent type
        """
        self.model_info = model_info
        self.score = 0

    def _call_api(self, prompt: str, state: AgentState):
        """
        Route the prompt to different API.

        Args:
            prompt (str): your prompt
            state (AgentState): the type of this prompt
        """
        response = None
        if self.model_info.name in OPENAI_LIST:
            response = self._call_openai(prompt, state)
        else:
            # TODO(wdxu): offline server
            response = self._call_offline()
        return response

    def _call_openai(self, prompt: str, state: AgentState):
        """
        Openai interface.

        Args:
            base_url (str): the base url of the openai server e.g. http://localhost:8000/v1
            prompt (str): your prompt
            state (AgentState): the type of this prompt
        """
        # For local inference, we don't need an API key
        api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
        client = OpenAI(
            api_key=api_key,
            base_url=self.model_info.base_url,
        )
        if state == AgentState.PATCH:
            response = client.chat.completions.create(
                model=self.model_info.name,
                # TODO(wdxu): need to designe a message passer.
                messages=[
                    {"role": "system", "content": GENERATE_PATCH_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
        else:
            response = client.chat.completions.create(
                model=self.model_info.name,
                messages=[
                    {"role": "system", "content": GENERATE_TEST_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
        return response

    def _call_offline(self):
        raise NotImplementedError("Offline server is not implemented yet.")

    def generate_patch(self, data: SwingbenchInstance, retriever: Retriever = None):
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
        prompt = GENERATE_PATCH_TEMPLATE.format(issue=issue, code_snippset=code_snippset)
        
        return self._call_api(prompt, AgentState.PATCH)

    def generate_test(self, data: SwingbenchInstance, retriever: Retriever):
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

        prompt = GENERATE_TEST_TEMPLATE.format(issue=issue, code_snippset=code_snippset, patch=patch, sample=sample)

        return self._call_api(prompt, AgentState.TEST)


if __name__ == "__main__":
    import swing_utils
    dataset_jsonl_path = '/mnt/Data/wdxu/github/Swing-Bench/tmpdata/dataset.json'
    dataset = swing_utils.load_swingbench_dataset(dataset_jsonl_path)
    index_dir = '/mnt/Data/wdxu/github/Swing-Bench/tmpdata/indexes'

    # model_info = ModelInfo(name="/home/mnt/wdxu/models/Qwen2.5-Coder-14B-Instruct", base_url="http://localhost:8000/v1")
    model_info = ModelInfo(name="/app/wdxu/models/Qwen2.5-Coder-32B", base_url="http://147.8.182.54:10000/v1")
    agent = AgentProxy(model_info)

    retriever = BM25DiskRetriever(index_dir=index_dir)

    for swing_instance in dataset:
        response = agent.generate_patch(swing_instance, retriever)
        print('patch reponse', response.choices[0].message.content)
        response = agent.generate_test(swing_instance, retriever)
        print('test reponse', response.choices[0].message.content)

        # results = retriever.retrieve(swing_instance)
        # print('retrieved instance id {} results {}'.format(swing_instance.instance_id, results))
