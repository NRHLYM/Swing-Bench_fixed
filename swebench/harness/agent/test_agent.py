import os
import logging
import json
import re

from swebench.harness.constants.swing_constants import SwingbenchInstance
from swebench.harness.agent.model import AgentProxy

class TestAgent:
    def __init__(self, 
                 model_name: str,
                 base_url: str = None,
                 api_key: str = None,
                 temperature: float = 0.0,
                 max_tokens: int = 2048,
                 top_p: float = 1.0,
                 workdir: str = "testbed", 
                 src_folder: str = "repos"):
        self.workdir = workdir
        self.src_folder = src_folder
        self.agent_proxy = AgentProxy(
            name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("TestAgent")
    
    def verify_patch(self, 
                     patch: str, 
                     testcase: str, 
                     problem_statement: str = None,
                     golden_patch: str = None,
                     use_problem: bool = False,
                     use_test: bool = True,
                     use_golden: bool = False,
                     data: SwingbenchInstance = None) -> dict:
        """
        Args:
            patch: Patch to verify (in git diff format)
            testcase: Test case to verify against (in git diff format)
            problem_statement: Problem statement (required if use_problem=True)
            golden_patch: Golden patch to compare against (required if use_golden=True)
            use_problem: Whether to use problem statement in verification
            use_test: Whether to use test case in verification (default: True)
            use_golden: Whether to use golden patch in verification
            data: SwingbenchInstance containing problem data (optional, can be used instead of problem_statement)

        Returns:
            Dict with verification results: {"success": bool, "confidence": str, "issues": list, "full_response": str}
        """
        if data and not problem_statement and use_problem:
            problem_statement = data.problem_statement
            if hasattr(data, 'hints_text') and data.hints_text:
                problem_statement += "\n" + data.hints_text
        
        assert not use_problem or (use_problem and problem_statement), "Problem statement is required when use_problem=True"
        assert not use_golden or (use_golden and golden_patch), "Golden patch is required when use_golden=True"
        assert use_test, "Test case is required (use_test cannot be False)"
        
        self.logger.info(f"Verifying patch with: use_problem={use_problem}, use_test={use_test}, use_golden={use_golden}")
        
        # Determine verification mode based on flags
        if use_problem and use_golden:
            result = self._verify_with_problem_test_and_golden(patch, testcase, problem_statement, golden_patch)
        elif use_problem:
            result = self._verify_with_problem_and_test(patch, testcase, problem_statement)
        else:
            result = self._verify_with_test_only(patch, testcase)
        
        return result
    
    def _verify_with_problem_and_test(self, patch: str, testcase: str, problem_statement: str) -> dict:
        system_prompt = """You are an expert code reviewer. 
Your task is to evaluate if a patch correctly solves a given problem based on the provided test case.
You will be given:
1. A problem statement
2. A test case
3. A patch that aims to solve the problem

Carefully analyze the test case and patch to determine if the patch correctly addresses the problem and passes the test case.
Provide a detailed reasoning for your conclusion.

Your response must be in JSON format with the following structure:
{
    "reasoning": "Detailed step-by-step analysis of how the patch addresses the problem and meets the test case requirements",
    "success": true/false (Does the patch correctly solve the problem according to the test case?),
    "confidence": "VERY_LOW" | "LOW" | "MEDIUM" | "HIGH" | "VERY_HIGH",
    "issues": [] (List of potential issues or limitations if any)
}

Confidence levels:
- VERY_LOW: Almost no certainty, unable to make a definitive judgment due to insufficient information or code complexity
- LOW: Some aspects seem correct, but major uncertainties remain
- MEDIUM: Reasonably confident in the assessment, but some minor doubts remain
- HIGH: Very confident in the assessment with only trivial uncertainties
- VERY_HIGH: Completely certain about the assessment with no doubts whatsoever
"""
        
        user_prompt = f"""## Problem Statement
{problem_statement}

## Test Case
```
{testcase}
```

## Patch
```
{patch}
```

Evaluate if this patch correctly solves the problem according to the test case. Return your analysis in the specified JSON format.
"""
        
        return self._get_model_verification(system_prompt, user_prompt)
    
    def _verify_with_test_only(self, patch: str, testcase: str) -> dict:
        system_prompt = """You are an expert code reviewer. 
Your task is to evaluate if a patch passes the provided test case.
You will be given:
1. A test case
2. A patch that aims to pass the test

Carefully analyze if the patch implementation correctly addresses the requirements outlined in the test case.
Provide a detailed reasoning for your conclusion.

Your response must be in JSON format with the following structure:
{
    "reasoning": "Detailed step-by-step analysis of how the patch meets the test case requirements",
    "success": true/false (Does the patch pass the test case?),
    "confidence": "VERY_LOW" | "LOW" | "MEDIUM" | "HIGH" | "VERY_HIGH",
    "issues": [] (List of potential issues or limitations if any)
}

Confidence levels:
- VERY_LOW: Almost no certainty, unable to make a definitive judgment due to insufficient information or code complexity
- LOW: Some aspects seem correct, but major uncertainties remain
- MEDIUM: Reasonably confident in the assessment, but some minor doubts remain
- HIGH: Very confident in the assessment with only trivial uncertainties
- VERY_HIGH: Completely certain about the assessment with no doubts whatsoever
"""
        
        user_prompt = f"""## Test Case
```
{testcase}
```

## Patch
```
{patch}
```

Evaluate if this patch successfully passes the test case. Return your analysis in the specified JSON format.
"""
        
        return self._get_model_verification(system_prompt, user_prompt)
    
    def _verify_with_problem_test_and_golden(self, patch: str, testcase: str, 
                                            problem_statement: str, golden_patch: str) -> dict:
        system_prompt = """You are an expert code reviewer. 
Your task is to evaluate if a patch correctly solves a given problem based on:
1. The problem statement
2. A test case
3. A reference "golden" patch known to correctly solve the problem

Compare the candidate patch with the golden patch to determine if they are functionally equivalent 
in terms of solving the problem and passing the test case.
Provide a detailed reasoning for your conclusion.

Your response must be in JSON format with the following structure:
{
    "reasoning": "Detailed step-by-step analysis comparing the candidate patch with the golden patch",
    "success": true/false (Is the candidate patch functionally equivalent to the golden patch?),
    "confidence": "VERY_LOW" | "LOW" | "MEDIUM" | "HIGH" | "VERY_HIGH",
    "issues": [] (List of differences or potential issues in the candidate patch)
}

Confidence levels:
- VERY_LOW: Almost no certainty, unable to make a definitive judgment due to insufficient information or code complexity
- LOW: Some aspects seem correct, but major uncertainties remain
- MEDIUM: Reasonably confident in the assessment, but some minor doubts remain
- HIGH: Very confident in the assessment with only trivial uncertainties
- VERY_HIGH: Completely certain about the assessment with no doubts whatsoever
"""
        
        user_prompt = f"""## Problem Statement
{problem_statement}

## Test Case
```
{testcase}
```

## Golden Patch (Known to be correct)
```
{golden_patch}
```

## Candidate Patch
```
{patch}
```

Evaluate if the candidate patch is functionally equivalent to the golden patch in solving the problem 
and passing the test case. Return your analysis in the specified JSON format.
"""
        
        return self._get_model_verification(system_prompt, user_prompt)
    
    def _get_model_verification(self, system_prompt: str, user_prompt: str) -> dict:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.agent_proxy.generate(messages)
            self.logger.info(f"Model response: {response}")
            
            try:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
                if json_match:
                    result_json = json.loads(json_match.group(1))
                else:
                    result_json = json.loads(response)
                
                success = result_json.get("success", False)
                reasoning = result_json.get("reasoning", "No reasoning provided")
                confidence = result_json.get("confidence", "MEDIUM")
                issues = result_json.get("issues", [])
                
                return {
                    "success": success,
                    "confidence": confidence,
                    "issues": issues,
                    "full_response": response
                }
                
            except json.JSONDecodeError:
                if "successfully" in response.lower() or "passes" in response.lower():
                    return {
                        "success": True,
                        "details": {"confidence": "LOW", "full_response": response}
                    }
                else:
                    return {
                        "success": False,
                        "details": {"confidence": "VERY_LOW", "full_response": response}
                    }
        
        except Exception as e:
            self.logger.error(f"Error during model verification: {str(e)}")
            return {
                "success": False,
                "details": {"error": str(e), "confidence": "VERY_LOW"}
            }


if __name__ == "__main__":
    sample_testcase = """--- /dev/null
+++ b/test_add.py
@@ -0,0 +1,15 @@
+import unittest
+from calculator import add
+
+class TestAdd(unittest.TestCase):
+    def test_add_positive(self):
+        self.assertEqual(add(1, 2), 3)
+        self.assertEqual(add(5, 7), 12)
+    
+    def test_add_negative(self):
+        self.assertEqual(add(-1, -2), -3)
+        self.assertEqual(add(-5, 7), 2)
+    
+if __name__ == '__main__':
+    unittest.main()
+"""
    
    correct_patch = """--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b  # Bug: subtraction instead of addition
+    return a + b  # Fixed: now correctly performs addition
"""
    
    incorrect_patch = """--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,3 @@
 def add(a, b):
-    return a - b  # Bug: subtraction instead of addition
+    # Still incorrect but with a comment change
+    return a - b
"""
    
    problem_statement = "Fix the bug in the calculator's add function. It's currently subtracting instead of adding."
    
    test_agent = TestAgent(
        model_name="qwen-max-latest",
        api_key="sk-826b874003eb4f309bd65c7a6f0f79b5",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/"
    )
    
    print("==== Testing with problem statement and test case (correct patch) ====")
    result = test_agent.verify_patch(
        patch=correct_patch,
        testcase=sample_testcase,
        problem_statement=problem_statement,
        use_problem=True
    )
    print(result)
    print(f"Success: {result['success']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Issues: {result['issues']}")
    print(f"Full response: {result['full_response']}")
    
    print("\n==== Testing with problem statement and test case (incorrect patch) ====")
    result = test_agent.verify_patch(
        patch=incorrect_patch,
        testcase=sample_testcase,
        problem_statement=problem_statement,
        use_problem=True
    )
    print(f"Success: {result['success']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Issues: {result['issues']}")
    print(f"Full response: {result['full_response']}")
    
    print("\n==== Testing with test case only (correct patch) ====")
    result = test_agent.verify_patch(
        patch=correct_patch,
        testcase=sample_testcase
    )
    print(f"Success: {result['success']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Issues: {result['issues']}")
    print(f"Full response: {result['full_response']}")
    
    print("\n==== Testing with problem, test, and golden patch ====")
    slightly_different_patch = """--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,3 @@
 def add(a, b):
-    return a - b  # Bug: subtraction instead of addition
+    # Fixed with a different comment
+    return a + b
"""
    result = test_agent.verify_patch(
        patch=slightly_different_patch,
        testcase=sample_testcase,
        problem_statement=problem_statement,
        golden_patch=correct_patch,
        use_problem=True,
        use_golden=True
    )
    print(f"Success: {result['success']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Issues: {result['issues']}")
    print(f"Full response: {result['full_response']}")