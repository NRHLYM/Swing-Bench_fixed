import os
from openai import OpenAI
from swebench.harness.constants.swing_constants import(
    AgentState,
    SwingbenchInstance
)

OPENAI_LIST = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4.5-preview"]

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


class AgentProxy:
    def __init__(self, name: str):
        """
        Initialize agent proxy.

        Args:
            name (str): agent type
        """
        self.name = name
        self.score = 0

    def call_api(self, prompt, state):
        """
        Route the prompt to different API.

        Args:
            prompt (str): your prompt
            state (AgentState): the type of this prompt
        """
        if self.name in OPENAI_LIST:
            self.call_openai(prompt, state)
        # TODO: offline server

    def call_openai(self, base_url: str, prompt: str, state: AgentState):
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
            base_url=base_url,
        )
        if state == AgentState.PATCH:
            response = client.chat.completions.create(
                model=self.name,
                messages=[
                    {"role": "developer", "content": GENERATE_PATCH_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
        else:
            response = client.chat.completions.create(
                model=self.name,
                messages=[
                    {"role": "developer", "content": GENERATE_TEST_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
        return response

    def generate_patch(self, data: SwingbenchInstance):
        """
        Patch generater.

        Args:
            data (SwingbenchInstance): a piece of data from dataset
        """
        issue = data.problem_statement + "\n" + data.hints_text
        code_snippset = data.related_code_snippset

        prompt = GENERATE_PATCH_TEMPLATE.format(issue=issue, code_snippset=code_snippset)
        
        return self.call_api(prompt, AgentState.PATCH)

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

        return self.call_api(prompt, AgentState.TEST)


if __name__ == "__main__":
    # http://localhost:8200/v1
    agent = AgentProxy("/home/mnt/wdxu/models/DeepSeek-R1-Distill-Qwen-7B")
    response = agent.call_openai("http://localhost:8200/v1", "Hello, world!", AgentState.PATCH)
    print(response)