GENERATE_PATCH_SYSTEM_MESSAGE = """
You are an AI Senior Full-Stack Engineer specialized in GitHub issue triage and bug fixing.
You should only generate the fixed code, without any other text or markdown formatting.
""".strip()

# TODO(wdxu): add testcase sample
TESTCASE_SAMPLE = """

"""

GENERATE_TEST_SYSTEM_MESSAGE = "You are an AI Test Automation Engineer specializing in generating unit tests." \
                                "You should only generate the test code, without any other text or markdown formatting."
GENERATE_TEST_TEMPLATE = "You are required to develop unit tests for the specified code and its fix.\n" \
                          "The issue details: {issue}\n" \
                          "The code snippet: {code_snippset}\n" \
                          "The fixed code: {patch}\n" \
                          "The test case sample: {sample}\n" \
                          "Please provide the complete test code without any explanations or markdown."

class Prompt:
    def __init__(self):
        self.system_message = None
        self.user_prompt = None


# TODO(haoran): better prompt
# TODO(haoran): supporting more files
# TODO(haoran): add naive function calling
swing_patch_system_prompt = "Analyze and modify code to resolve issues while preserving functionality. You should use code_editor to process the intput field information. You should use <response>...</response> to wrap the code_editor output."
swing_patch_retry_prompt = "The previous response is not correct because it is not a valid json object, please try again. The previous response is: "
swing_patch_function = {
    "name": "code_editor",
    "description": f"{swing_patch_system_prompt}",
    "parameters": {
            "type": "object",
            "properties": {
                "reasoning_trace": {
                    "type": "string",
                    "description": "Step-by-step analysis of the issue, explanation of the root cause, and justification for the proposed solution. Do not use any markdown formatting."
                },
                "code_edits": {
                    "type": "array",
                    "description": "List of specific code modifications required to resolve the issue",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "description": "Relative path to the file that contains code requiring modification"
                            },
                            "code_to_be_modified": {
                                "type": "string",
                                "description": "Exact code segment that needs to be changed (must match a portion of the original file)"
                            },
                            "code_edited": {
                                "type": "string",
                                "description": "Improved version of the code segment that fixes the issue while maintaining compatibility with surrounding code"
                            }
                        },
                        "required": ["file", "code_to_be_modified", "code_edited"]
                    }
                }
            },
        "required": ["reasoning_trace", "code_edits"]
    }
}


swing_test_system_prompt = "You are an AI Test Automation Engineer specializing in generating comprehensive unit tests. Your task is to analyze the provided code and create effective test cases that verify the functionality and edge cases. You should use <response>...</response> to wrap your JSON output. For convinience, you should not create new test files, only modify the existing test files."
swing_test_retry_prompt = "The previous response is not correct because it is not a valid json object, please try again. The previous response is: "
swing_test_function = {
    "name": "test_generator",
    "description": f"{swing_test_system_prompt}",
    "parameters": {
            "type": "object",
            "properties": {
                "reasoning_trace": {
                    "type": "string",
                    "description": "Step-by-step analysis of the code, explanation of what needs to be tested, and justification for the test cases. Do not use any markdown formatting."
                },
                "test_cases": {
                    "type": "array",
                    "description": "List of test cases to verify the functionality of the code",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "description": "Relative path to the test file where the test case should be added"
                            },
                            "original_patch": {
                                "type": "string",
                                "description": "The original patch that provided by generator. You should not modify this patch but should investigate the existing test files and create test cases."
                            },
                            "test_name": {
                                "type": "string",
                                "description": "Descriptive name of the test case"
                            },
                            "test_code": {
                                "type": "string",
                                "description": "Complete test code including setup, execution, and assertions"
                            },
                            "test_description": {
                                "type": "string",
                                "description": "Brief description of what the test case verifies"
                            }
                        },
                        "required": ["file", "test_name", "test_code", "test_description"]
                    }
                }
            },
            "required": ["reasoning_trace", "test_cases"]
        }
    }
