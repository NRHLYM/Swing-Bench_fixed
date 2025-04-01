import os
from openai import OpenAI
from swebench.harness.constants.swing_constants import(
    AgentState
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
                          "The patch: {patch} " \
                          "The test case sample: {sample} "

class Retriever:
    def __init__(self):
        pass

    def retrieve(self, codebase, query):
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

    def verify_patch(self, data, patch, test=None):
        """
        Patch verifier.

        Args:
            data (dict): a piece of data from dataset
            patch: patch string? temporary patch file?
            test: test string? temporary test file?
        
        """
        pass

    def verify_test(self, data, test):
        """
        Test verifier.

        Args:
            data (dict): a piece of data from dataset
            test: test string? temporary test file?
        """
        pass

class AgentProxy:
    def __init__(self, name):
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

    def call_openai(self, prompt, state):
        """
        Openai interface.

        Args:
            prompt (str): your prompt
            state (AgentState): the type of this prompt
        """
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Must provide an api key. Expected in OPENAI_API_KEY environment variable.")
        client = OpenAI()
        if state == AgentState.PATCH:
            response = client.responses.create(
                model=self.name,
                input=[
                    {"role": "developer", "content": GENERATE_PATCH_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
        else:
            response = client.responses.create(
                model=self.name,
                input=[
                    {"role": "developer", "content": GENERATE_TEST_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
        
    def generate_patch(self, data):
        """
        Patch generater.

        Args:
            data (dict): a piece of data from dataset
        """
        issue = None
        code_snippset = None
        prompt = None
        return self.call_api(prompt, AgentState.PATCH)

    def generate_test(self, data):
        """
        Test generater.

        Args:
            data (str): a piece of data from dataset
        """
        issue = None
        patch = None
        prompt = None
        return self.call_api(prompt, AgentState.TEST)