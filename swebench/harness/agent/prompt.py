GENERATE_PATCH_SYSTEM_MESSAGE = "You are an AI Senior Full-Stack Engineer specialized in GitHub issue triage and bug fixing." \
                                "You should only generate the fixed code, without any other text or markdown formatting."
GENERATE_PATCH_TEMPLATE = "You are required to fix the code for the specified issue.\n" \
                          "The issue details: {issue}\n" \
                          "The code snippet: {code_snippset}\n" \
                          "Please provide the complete fixed code without any explanations or markdown."

GENERATE_TEST_SYSTEM_MESSAGE = "You are an AI Test Automation Engineer specializing in generating unit tests." \
                                "You should only generate the test code, without any other text or markdown formatting."
GENERATE_TEST_TEMPLATE = "You are required to develop unit tests for the specified code and its fix.\n" \
                          "The issue details: {issue}\n" \
                          "The code snippet: {code_snippset}\n" \
                          "The fixed code: {patch}\n" \
                          "The test case sample: {sample}\n" \
                          "Please provide the complete test code without any explanations or markdown."
