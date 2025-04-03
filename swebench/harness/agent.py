import os
from openai import OpenAI
from swebench.harness.constants.swing_constants import(
    AgentState,
    SwingbenchInstance
)
from swebench.inference.make_datasets.bm25_retrieval import run_bm25

OPENAI_LIST = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4.5-preview",
               "/home/mnt/wdxu/models/DeepSeek-R1-Distill-Qwen-7B"]

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
GENERATE_PATCH_SYSTEM_MESSAGE = "You are an AI Senior Full-Stack Engineer specialized in GitHub issue triage and bug fixing."
GENERATE_PATCH_TEMPLATE = "You are required to write a patch for the specified issue and its corresponding code section. " \
                          "The issue details: {issue} " \
                          "The code snippet: {code_snippset} "
# change to a more efficient template
GENERATE_TEST_SYSTEM_MESSAGE = "You are an AI Test Automation Engineer specializing in generating unit tests for code patches."
GENERATE_TEST_TEMPLATE = "You are required to develop unit tests for the specified patch, which was created to resolve this issue. " \
                          "The issue details: {issue} " \
                          "The code snippet: {code_snippset} " \
                          "The patch: {patch} " \
                          "The test case sample: {sample} "

class Retriever:
    def __init__(self):
        pass

    def retrieve(self, codebase: str, query: str):
        """
        Retrieve target code snippset from codebase.

        Args:
            codebase: 
            query: 

        """
        pass

    def bm25_retrieve(self, dataset_name_or_path: str, file_name: str, document_encoding_style: str, output_dir: str):
        """
        BM25 retrieve.

        Args:
            dataset_name_or_path: dataset to use for test set from HuggingFace Datasets or path to a save_to_disk directory
            file_name: temporary data file to debug
            document_encoding_style: the function to build a document
            output_dir: the output directory to save the retrieved results

        """
        run_bm25(dataset_name_or_path, file_name, document_encoding_style, output_dir)


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
                messages=[
                    {"role": "developer", "content": GENERATE_PATCH_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
        else:
            response = client.chat.completions.create(
                model=self.model_info.name,
                messages=[
                    {"role": "developer", "content": GENERATE_TEST_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
        return response

    def _call_offline(self):
        raise NotImplementedError("Offline server is not implemented yet.")

    def generate_patch(self, data: SwingbenchInstance):
        """
        Patch generater.

        Args:
            data (SwingbenchInstance): a piece of data from dataset
        """
        issue = data.problem_statement + "\n" + data.hints_text
        code_snippset = data.related_code_snippset

        prompt = GENERATE_PATCH_TEMPLATE.format(issue=issue, code_snippset=code_snippset)
        
        return self._call_api(prompt, AgentState.PATCH)

    def generate_test(self, data: SwingbenchInstance):
        """
        Test generater.

        Args:
            data (SwingbenchInstance): a piece of data from dataset
        """
        issue = data.problem_statement + "\n" + data.hints_text
        code_snippset = data.related_code_snippset
        patch = data.patch
        sample = data.test_patch

        prompt = GENERATE_TEST_TEMPLATE.format(issue=issue, code_snippset=code_snippset, patch=patch, sample=sample)

        return self._call_api(prompt, AgentState.TEST)


if __name__ == "__main__":
    import swing_utils
    dataset_jsonl_path = '/mnt/Data/wdxu/github/Swing-Bench/tmpdata/dataset.json'
    dataset = swing_utils.load_swingbench_dataset(dataset_jsonl_path)

    model_info = ModelInfo(name="/home/mnt/wdxu/models/DeepSeek-R1-Distill-Qwen-7B", base_url="http://localhost:8200/v1")
    agent = AgentProxy(model_info)

    for swing_instance in dataset:
        response = agent.generate_patch(swing_instance)
        print('patch reponse', response.choices[0].message.content)
        response = agent.generate_test(swing_instance)
        print('test reponse', response.choices[0].message.content)
