import os
import sys

# 设置临时目录 - 在最开始设置
temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "temp")
os.makedirs(temp_dir, exist_ok=True)
os.environ['TMPDIR'] = temp_dir
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir

import copy
from abc import abstractmethod
from datetime import datetime
import subprocess
from uuid import uuid4
from swebench.harness.constants.swing_constants import SwingbenchInstance
from swebench.harness.router import CIToolBase
from swebench.harness.router import EVAL_HANDLER
from swebench.harness.agent.model import AgentProxy
import shutil
import tempfile
# 强制设置临时目录
tempfile.tempdir = temp_dir

from swebench.harness.agent.editor import generate_git_diff_batch
from swebench.harness.agent.retriever import Retriever
from swebench.harness.agent.editor import CodeEditorBase
from swebench.harness.utils import get_available_port_pool
from swebench.harness.swing_utils import merge_diffs

from tree_sitter import Language, Parser
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Any
import re

class CodeReranker:
    def __init__(self, model_name="microsoft/codebert-base"):
        """
        初始化代码重排序器
        
        Args:
            model_name: 用于计算embedding的模型名称
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            print(f"CodeReranker initialized with {model_name} on {self.device}")
            self.initialized = True
        except Exception as e:
            print(f"Failed to initialize CodeReranker: {e}")
            self.initialized = False
    
    def get_embeddings(self, code_texts):
        """计算代码文本的嵌入向量"""
        if not self.initialized:
            return None
            
        try:
            # 使用tokenizer处理代码文本
            inputs = self.tokenizer(code_texts, padding=True, truncation=True, 
                                    max_length=512, return_tensors="pt").to(self.device)
            
            # 不计算梯度
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 使用[CLS]标记的embedding作为整个序列的表示
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            # 转换为numpy数组并返回
            return embeddings.cpu().numpy()
        except Exception as e:
            print(f"Error during embedding calculation: {e}")
            return None
    
    def calculate_similarity(self, query_embedding, code_embeddings):
        """计算查询和代码块之间的余弦相似度"""
        if not self.initialized:
            return None
            
        # 对embeddings进行正则化
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        code_norms = code_embeddings / np.linalg.norm(code_embeddings, axis=1, keepdims=True)
        
        # 计算余弦相似度
        similarities = np.dot(code_norms, query_norm.T).flatten()
        return similarities
    
    def rerank(self, chunks, problem_statement, top_k=3):
        """
        根据与问题描述的相关性对代码块进行重新排序
        
        Args:
            chunks: 代码块列表
            problem_statement: 问题描述
            top_k: 返回的顶部代码块数量
            
        Returns:
            按相关性排序的前top_k个代码块
        """
        if not self.initialized or not chunks:
            return chunks[:top_k] if chunks else []
        
        # 提取代码文本
        code_texts = [chunk["code"] for chunk in chunks]
        
        # 获取问题和代码的嵌入
        try:
            all_texts = [problem_statement] + code_texts
            all_embeddings = self.get_embeddings(all_texts)
            
            if all_embeddings is None:
                return chunks[:top_k]
                
            # 分离问题和代码的嵌入
            query_embedding = all_embeddings[0]
            code_embeddings = all_embeddings[1:]
            
            # 计算相似度
            similarities = self.calculate_similarity(query_embedding, code_embeddings)
            
            # 获取排序后的索引
            sorted_indices = np.argsort(-similarities)  # 降序排序
            
            # 选择top_k个代码块
            top_indices = sorted_indices[:min(top_k, len(chunks))]
            reranked_chunks = [chunks[i] for i in top_indices]
            
            # 添加相似度分数到代码块中
            for i, chunk in enumerate(reranked_chunks):
                chunk["similarity_score"] = float(similarities[top_indices[i]])
                
            return reranked_chunks
        except Exception as e:
            print(f"Error during reranking: {e}")
            return chunks[:top_k]

class Verifier:
    @abstractmethod
    def __init__(self, ci_tool: CIToolBase):
        raise NotImplementedError

    @abstractmethod
    def _extract_code(self, input: str):
        raise NotImplementedError

    @abstractmethod
    def verify(self, data: SwingbenchInstance, input: str):
        raise NotImplementedError


class Generator:
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def generate(self, data: SwingbenchInstance):
        raise NotImplementedError

class CodeChunker:
    def __init__(self, language: str = "rust", chunk_type: str = "function"):
    
        self.language = language
        self.chunk_type = chunk_type
        
 
        if language == "python":
            import tree_sitter_python
            PYTHON_LANGUAGE = Language(tree_sitter_python.language())
            self.parser = Parser(PYTHON_LANGUAGE)
            
            print(f"Successfully loaded Python language: {self.parser}")

                
        elif language == "javascript":
            import tree_sitter_javascript
            JS_LANGUAGE = Language(tree_sitter_javascript.language())
            self.parser = Parser(JS_LANGUAGE)
            
            print(f"Successfully loaded JavaScript language: {self.parser}")

                
        elif language == "typescript":
            import tree_sitter_typescript
            TS_LANGUAGE = Language(tree_sitter_typescript.language())
            self.parser = Parser(TS_LANGUAGE)
    
            print(f"Successfully loaded TypeScript language: {self.parser}")
        
        elif language == "rust":
            import tree_sitter_rust as tsrust
            RUST_LANGUAGE = Language(tsrust.language())
            self.parser = Parser(RUST_LANGUAGE)
        
            print(f"Successfully loaded Rust language: {self.parser}")
                
        else:
            print(f"Language {language} not supported, will use regex to parse")
            self.parser = None
                

    def chunk(self, code_snippet: str = None) -> List[Dict[str, Any]]:
        """
        Chunk the code snippet into smaller, semantically meaningful pieces.
        
        Args:
            chunk_type: The type of chunks to extract. Options: "function", "class", "block"
        
        Returns:
            A list of dictionaries containing chunk information:
            [
                {
                    "type": str,  # Type of the chunk (function, class, etc.)
                    "name": str,  # Name of the function, class, etc.
                    "code": str,  # The code content of the chunk
                    "start_line": int,  # Starting line number
                    "end_line": int,  # Ending line number
                    "metadata": dict  # Additional metadata about the chunk
                },
                ...
            ]
        """
        print("start chunk in here==========================:{}".format(code_snippet))
        
        if not code_snippet or code_snippet.strip() == "":
            return []
            
        # If tree-sitter parser is available, use it
        if self.parser is not None:
                print("start _chunk_with_tree_sitter in here==========================")
                #assert 1==0
                return self._chunk_with_tree_sitter(code_snippet)

            
        else:
            #assert 1==0
            # Use regex-based chunking as fallback
            return self._chunk_with_regex(code_snippet)
    
    def _chunk_with_tree_sitter(self,  code_snippet: str = None) -> List[Dict[str, Any]]:
        """Use tree-sitter to extract code chunks."""
        print("start _chunk_with_tree_sitter in here==========================")
        chunks = []
        tree = self.parser.parse(bytes(code_snippet, "utf8"))
        root_node = tree.root_node
        
        # Split code into lines for line number references
        #lines = code_snippet.split("\n")
        
        # Define query patterns based on chunk_type and language
        if self.language == "python":
            if self.chunk_type == "function":
                query_string = "(function_definition name: (identifier) @func_name) @function"
            elif self.chunk_type == "class":
                query_string = "(class_definition name: (identifier) @class_name) @class"
            elif self.chunk_type == "block":
                query_string = """
                (function_definition) @block
                (class_definition) @block
                (if_statement) @block
                (for_statement) @block
                (while_statement) @block
                (try_statement) @block
                """
            else:
                query_string = "(function_definition) @function (class_definition) @class"
        elif self.language == "rust":
            print("self.language:{}".format(self.language))
            # Use the correct Rust syntax node type
            if self.chunk_type == "function":
                query_string = "(function_item name: (identifier) @func_name) @function"
            elif self.chunk_type == "class" or self.chunk_type == "struct":
                query_string = "(struct_item name: (type_identifier) @struct_name) @struct"
            elif self.chunk_type == "block":
                query_string = """
                (function_item) @block
                (struct_item) @block
                (impl_item) @block
                (trait_item) @block
                (if_expression) @block
                (for_expression) @block
                (while_expression) @block
                """
            else:
                query_string = """
                (function_item) @function 
                (struct_item) @struct
                (impl_item) @impl
                """
        else:
            # Default query patterns for other languages
            if self.chunk_type == "function":
                query_string = "(function_declaration name: (identifier) @func_name) @function"
            elif self.chunk_type == "class":
                query_string = "(class_declaration name: (identifier) @class_name) @class"
            else:
                query_string = "(function_declaration) @function (class_declaration) @class"
        
        # Create and execute the query
        language = self.parser.language
        query = language.query(query_string)
        captures = query.captures(root_node)
        
        # 新版tree-sitter返回的是字典结构，每个键是标签名称，值是匹配的节点列表
        # 正确处理这种字典结构
        # 我们只关注主要节点（function, class, struct等），而不是它们的子节点（如func_name）
        main_tags = ["function", "class", "struct", "block", "impl"]
        
        # 遍历每个主要标签
        for tag in main_tags:
            if tag in captures:
                # 处理这个标签下的所有节点
                for node in captures[tag]:
                    
                    start_point = node.start_point
                    end_point = node.end_point
                    start_line = start_point[0]
                    end_line = end_point[0]
                    
                    # 获取节点文本
                    node_text = code_snippet[node.start_byte:node.end_byte]
                    
                    # 尝试提取名称
                    name = ""
                    # 看是否有对应的name节点
                    name_tag = None
                    if tag == "function":
                        name_tag = "func_name"
                    elif tag == "class":
                        name_tag = "class_name"
                    elif tag == "struct":
                        name_tag = "struct_name"
                    
                    # 如果有对应的name标签，直接从那里获取名称
                    if name_tag and name_tag in captures:
                        # 需要找到对应于当前节点的name节点
                        for name_node in captures[name_tag]:
                            # 检查name_node是否是当前node的子节点
                            if name_node.start_byte >= node.start_byte and name_node.end_byte <= node.end_byte:
                                name = code_snippet[name_node.start_byte:name_node.end_byte]
                                break
                    
                    # 如果没有从name标签中获取到名称，尝试从子节点中查找
                    if not name:
                        for child in node.children:
                            if child.type == "identifier" or child.type == "type_identifier":
                                name = code_snippet[child.start_byte:child.end_byte]
                                break
                    
                    chunks.append({
                        "type": tag,
                        "name": name,
                        "code": node_text,
                        "start_line": start_line + 1,  # 1-indexed line numbers
                        "end_line": end_line + 1,
                        "metadata": {
                            "node_type": node.type,
                            "byte_range": (node.start_byte, node.end_byte)
                        }
                    })
        
        return chunks
    
    def _chunk_with_regex(self, code_snippet: str = None) -> List[Dict[str, Any]]:
        """Use regex patterns to extract code chunks as fallback."""
        chunks = []
        lines = code_snippet.split("\n")
        
        if self.language == "python":
            if self.chunk_type in ["function", "block"]:
                # Match function definitions
                pattern = r"^(\s*)def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
                matches = []
                for i, line in enumerate(lines):
                    match = re.match(pattern, line)
                    if match:
                        matches.append((i, match.group(1), match.group(2)))
                
                # Process matches to get function chunks
                for j, (start_idx, indent, func_name) in enumerate(matches):
                    # Find where function ends (next line with same or less indentation)
                    end_idx = start_idx
                    for k in range(start_idx + 1, len(lines)):
                        if lines[k].strip() and not lines[k].startswith(indent + " "):
                            end_idx = k - 1
                            break
                        end_idx = k
                    
                    chunks.append({
                        "type": "function",
                        "name": func_name,
                        "code": "\n".join(lines[start_idx:end_idx+1]),
                        "start_line": start_idx + 1,
                        "end_line": end_idx + 1,
                        "metadata": {"indent": len(indent)}
                    })
            
            if self.chunk_type in ["class", "block"]:
                # Match class definitions
                pattern = r"^(\s*)class\s+([a-zA-Z_][a-zA-Z0-9_]*)"
                matches = []
                for i, line in enumerate(lines):
                    match = re.match(pattern, line)
                    if match:
                        matches.append((i, match.group(1), match.group(2)))
                
                # Process matches to get class chunks
                for j, (start_idx, indent, class_name) in enumerate(matches):
                    # Find where class ends
                    end_idx = start_idx
                    for k in range(start_idx + 1, len(lines)):
                        if lines[k].strip() and not lines[k].startswith(indent + " "):
                            end_idx = k - 1
                            break
                        end_idx = k
                    
                    chunks.append({
                        "type": "class",
                        "name": class_name,
                        "code": "\n".join(lines[start_idx:end_idx+1]),
                        "start_line": start_idx + 1,
                        "end_line": end_idx + 1,
                        "metadata": {"indent": len(indent)}
                    })
        
        return chunks

class PatchGenerator(Generator):
    def __init__(self, workdir: str = "testbed", 
                 src_folder: str = "repos",
                 code_editor: CodeEditorBase = None,
                 retriever: Retriever = None,
                 retrieve_file_num: int = 20,
                 agent_retry_times: int = 3,
                 max_chunk_num: int = 3,
                 ):
        self.workdir = workdir
        self.src_folder = src_folder
        self.code_editor = code_editor
        self.retriever = retriever
        self.retrieve_file_num = retrieve_file_num
        self.agent_retry_times = agent_retry_times
        self.chunker = CodeChunker(language="rust", chunk_type="function")
        self.max_chunk_num = max_chunk_num
        self.reranker = CodeReranker()
    def model_name(self):
        return self.code_editor.model

    def generate(self, data: SwingbenchInstance):
        """
        Generate a patch for the given Swingbench instance.
        """
        print("Retrieving data files from retriever to data:{}".format(data))
        code_snippet = self.retriever.retrieve(data, k=self.retrieve_file_num)
        print(f"self.retrieve_file_num: {self.retrieve_file_num}")
        print(f"len of code_snippet: {len(code_snippet)}")
        print(f"code_snippet.keys: {code_snippet.keys()}")
        print(f"len(code_snippet['hits']): {len(code_snippet['hits'])}")
        print(f"code_snippet[hits].keys: {code_snippet['hits'][0].keys()}")
        print(f"code_snippet[hits][0]: {code_snippet['hits'][0]}")
        print(f"code_snippet[hits][0]['docid']: {code_snippet['hits'][0]['docid']}")
        
        all_chunks = []
        # Chunk the code snippet for each hit
        for hit in code_snippet['hits']:
            # Use self.chunker to chunk the code
            chunks = self.chunker.chunk(code_snippet=hit['contents'])
            # Add the chunk results to the hit and associate the original file path
            for chunk in chunks:
                chunk['file_path'] = hit['docid']
            hit['chunks'] = chunks
            all_chunks.extend(chunks)
            # Print the chunking information 
            print(f"File {hit['docid']} has {len(chunks)} code chunks")
 
        # Rerank the all code chunks
        print(f"Total chunks before reranking: {len(all_chunks)}")
        if all_chunks and self.reranker.initialized:
            top_chunks = self.reranker.rerank(all_chunks, data.problem_statement, top_k=self.max_chunk_num)
            print(f"After reranking, selected top {len(top_chunks)} chunks:")
            for i, chunk in enumerate(top_chunks):
                print(f"  Top Chunk {i}: {chunk['type']} - {chunk['name']} (score: {chunk.get('similarity_score', 'N/A')})")
                print(f"    From file: {chunk['file_path']}")
        
        # Build the new input based on the reranked code chunks
        chunk_file_path_list = [chunk['file_path'] for chunk in top_chunks]
        chunk_code_list = [chunk['code'] for chunk in top_chunks]
        
        # Build the metadata information related to the code chunks to help the model understand the context
        context_info = []
        for chunk in top_chunks:
            context_info.append(f"File: {chunk['file_path']}\n"
                              f"Type: {chunk['type']}\n"
                              f"Name: {chunk['name']}\n"
                              f"Lines: {chunk['start_line']}-{chunk['end_line']}\n"
                              f"Code:\n{chunk['code']}\n")
        
        # Add the context information to the problem statement
        enhanced_problem = data.problem_statement + "\n\n" + "RELEVANT CODE BLOCKS:\n" + "\n".join(context_info)
        
        # Use the original file list as a backup, and provide the code chunks
        file_path_list = [hit["docid"] for hit in code_snippet["hits"]]
        code_snippet_list = [hit["contents"] for hit in code_snippet["hits"]]
        
        # Call the editor with the enhanced problem description, and provide the original file and the code chunks
        response = self.code_editor.edit_code_batch(
            enhanced_problem,
            code_snippet_list,  # Original complete code
            file_path_list,     # Original file path
            role="patch",
            retry=self.agent_retry_times,
            chunks={ # Add the code chunks information as additional context
                "file_paths": chunk_file_path_list,
                "code_blocks": chunk_code_list,
                "metadata": top_chunks
            }
        )
        if response is None:
            return None
        base_path = f"{self.workdir}/{data.instance_id}_{str(uuid4())}"

        # convert repo path from x/y to x__y
        repo_path = f"{self.src_folder}/{data.repo.replace('/', '__')}"

        if os.path.exists(base_path):
            # remove existing repo
            shutil.rmtree(base_path)

        shutil.copytree(repo_path, base_path)
        subprocess.run(["git", "checkout", data.base_commit], cwd=base_path)

        patch = generate_git_diff_batch(response["code_edits"], base_path)
        # if os.path.exists(base_path):
        #     shutil.rmtree(base_path)
        return patch


class TestGenerator(Generator):
    def __init__(self, workdir: str = "testbed", 
                 src_folder: str = "repos",
                 code_editor: CodeEditorBase = None,
                 retriever: Retriever = None,
                 retrieve_file_num: int = 20,
                 agent_retry_times: int = 3,
                 max_chunk_num: int = 3,
                 ):
        self.workdir = workdir
        self.src_folder = src_folder
        self.code_editor = code_editor
        self.retriever = retriever
        self.retrieve_file_num = retrieve_file_num
        self.agent_retry_times = agent_retry_times
        self.chunker = CodeChunker(language="rust", chunk_type="function")
        self.max_chunk_num = max_chunk_num
        self.reranker = CodeReranker()
    
    def model_name(self):
        return self.code_editor.model

    def generate(self, data: SwingbenchInstance, generated_patch: str = None):
        # TODO(wdxu): remove this hack.
        data.hints_text += "test, testcase, unittest."
        code_snippet = self.retriever.retrieve(data, k=self.retrieve_file_num)
        print(f"self.retrieve_file_num: {self.retrieve_file_num}")
        print(f"len of code_snippet: {len(code_snippet)}")
        print(f"code_snippet.keys: {code_snippet.keys()}")
        print(f"code_snippet[hits].keys: {code_snippet['hits'][0].keys()}")
        print(f"code_snippet[hits][0]: {code_snippet['hits'][0]}")
        
        all_chunks = []
        # Chunk the code snippet for each hit
        for hit in code_snippet['hits']:
            # Use self.chunker to chunk the code
            chunks = self.chunker.chunk(code_snippet=hit['contents'])
            # Add the chunk results to the hit and associate the original file path
            for chunk in chunks:
                chunk['file_path'] = hit['docid']
            hit['chunks'] = chunks
            all_chunks.extend(chunks)
            # Print the chunking information 
            print(f"File {hit['docid']} has {len(chunks)} code chunks")
        
        # Rerank the all code chunks
        print(f"Total chunks before reranking: {len(all_chunks)}")
        if all_chunks and self.reranker.initialized:
            top_chunks = self.reranker.rerank(all_chunks, data.problem_statement, top_k=self.max_chunk_num)
            print(f"After reranking, selected top {len(top_chunks)} chunks:")
            for i, chunk in enumerate(top_chunks):
                print(f"  Top Chunk {i}: {chunk['type']} - {chunk['name']} (score: {chunk.get('similarity_score', 'N/A')})")
                print(f"    From file: {chunk['file_path']}")
        
        # Build the new input based on the reranked code chunks
        chunk_file_path_list = [chunk['file_path'] for chunk in top_chunks]
        chunk_code_list = [chunk['code'] for chunk in top_chunks]
        
        # Build the metadata information related to the code chunks to help the model understand the context
        context_info = []
        for chunk in top_chunks:
            context_info.append(f"File: {chunk['file_path']}\n"
                              f"Type: {chunk['type']}\n"
                              f"Name: {chunk['name']}\n"
                              f"Lines: {chunk['start_line']}-{chunk['end_line']}\n"
                              f"Code:\n{chunk['code']}\n")
        
        # Add the context information and patch to the problem statement
        enhanced_problem = data.problem_statement + "\n\n" + "RELEVANT CODE BLOCKS:\n" + "\n".join(context_info)
        # print(f"generated_patch: {generated_patch}")
        # if generated_patch:
        #     enhanced_problem += "\n\nGENERATED PATCH:\n" + generated_patch
        
        # Use the original file list as a backup, and provide the code chunks
        file_path_list = [hit["docid"] for hit in code_snippet["hits"]]
        code_snippet_list = [hit["contents"] for hit in code_snippet["hits"]]
        
        # Call the editor with the enhanced problem description
        response = self.code_editor.edit_code_batch(
            enhanced_problem,
            code_snippet_list,  # Original complete code
            file_path_list,     # Original file path
            role="test",
            retry=self.agent_retry_times,
            generated_patch=generated_patch,
            chunks={ # Add the code chunks information as additional context
                "file_paths": chunk_file_path_list,
                "code_blocks": chunk_code_list,
                "metadata": top_chunks
            }
        )
        if response is None:
            return None
        base_path = f"{self.workdir}/{data.instance_id}_{str(uuid4())}"

        # convert repo path from x/y to x__y
        repo_path = f"{self.src_folder}/{data.repo.replace('/', '__')}"

        if os.path.exists(base_path):
            # remove existing repo
            shutil.rmtree(base_path)

        shutil.copytree(repo_path, base_path)
        subprocess.run(["git", "checkout", data.base_commit], cwd=base_path)
    
        try:
            patch = generate_git_diff_batch(response["test_cases"], base_path)
        except Exception as e:
            print(f"Error generating test cases: {e}")
            print(f"Response: {response}")
            print(f"File path list: {file_path_list}")
            print(f"Code snippet list: {code_snippet_list}")
            return None
        # if os.path.exists(base_path):
        #     shutil.rmtree(base_path)
        return patch


class PatchVerifier(Verifier):
    def __init__(self, ci_tool_name: str, 
                 workdir: str = "testbed", 
                 src_folder: str = "repos",
                 ):
        self.ci_tool_name = ci_tool_name
        self.workdir = workdir
        self.src_folder = src_folder
        self.port_pool_size = 100

    def verify(self, data: SwingbenchInstance, patch: str) -> dict:
        data.patch = patch
        base_path = f"{self.workdir}/{data.instance_id}_{str(uuid4())}"
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
        
        # CITool will handle the patch
        config = {
            "instance_id": data.instance_id,
            "repo": data.repo,
            "base_commit": data.base_commit,
            "merge_commit": data.merge_commit_sha,
            "patch": patch,
            "src_folder": self.src_folder,
            "output_dir": "logs",
            "workdir": base_path,
            "apply_patch": True,
            "ci_name_list": data.ci_name_list
        }
        ci_tool = EVAL_HANDLER.get(self.ci_tool_name)
        tool = ci_tool(config)

        # TODO(wdxu): remove the switch process for run_ci.
        if self.ci_tool_name == "act":
            pool = get_available_port_pool(self.port_pool_size)
            result = tool.run_ci(pool)
        else:
            result = tool.run_ci()

        # haoran: FOR CARGO
        # test_results = {
        #     "unit_test": {
        #         "passed": passed_tests,
        #         "failed": failed_tests,
        #         "ignored": ignored_tests,
        #         "failure_details": {}
        #     }
        # }
        # FOR ACT
        # test_results = {
        #     "ci_1": {
        #         "passed": passed_tests,
        #         "failed": failed_tests,
        #         "ignored": ignored_tests,
        #         "failure_details": {}
        #     }, ...
        # }

        return {
            "tool": self.ci_tool_name,
            "result": result,
            "patch": patch
        }


class TestVerifier(Verifier):
    def __init__(self, ci_tool_name: str, 
                 workdir: str = "testbed", 
                 src_folder: str = "repos",
                 proxy: AgentProxy = None,
                 ):
        self.ci_tool_name = ci_tool_name
        self.workdir = workdir
        self.src_folder = src_folder
        self.proxy = proxy
        self.port_pool_size = 100

    def verify(self, data: SwingbenchInstance, testcase: str) -> dict:
        # apply both test patch and original patch
        base_path = f"{self.workdir}/{data.instance_id}_{str(uuid4())}"
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
        
        config = {
            "instance_id": data.instance_id,
            "repo": data.repo,
            "base_commit": data.base_commit,
            "merge_commit": data.merge_commit_sha,
            "patch": testcase,
            "src_folder": self.src_folder,
            "output_dir": "logs",
            "workdir": base_path,
            "apply_patch": True,
            "ci_name_list": data.ci_name_list
        }
        ci_tool = EVAL_HANDLER.get(self.ci_tool_name)
        tool = ci_tool(config)

        # TODO(wdxu): remove the switch process for run_ci.
        if self.ci_tool_name == "act":
            pool = get_available_port_pool(self.port_pool_size)
            result = tool.run_ci(pool)
        else:
            result = tool.run_ci()

        # haoran: FOR CARGO
        # test_results = {
        #     "unit_test": {
        #         "passed": passed_tests,
        #         "failed": failed_tests,
        #         "ignored": ignored_tests,
        #         "failure_details": {}
        #     }
        # }
        # FOR ACT
        # test_results = {
        #     "ci_1": {
        #         "passed": passed_tests,
        #         "failed": failed_tests,
        #         "ignored": ignored_tests,
        #         "failure_details": {}
        #     }, ...
        # }

        return {
            "tool": self.ci_tool_name,
            "result": result,
            "test_cases": testcase
        }


if __name__ == "__main__":
    from swebench.harness.swing_utils import load_swingbench_dataset
    from swebench.harness.agent.retriever import BM25DiskRetriever
    from swebench.harness.agent.editor import RawDataCodeEditor
    from swebench.harness.agent.model import AgentProxy
    import json
    
    SWING_DEBUG_GENERATE_DRYRUN = False
    
    # base_url = "https://api.x.ai/v1/"
    # api_key = os.environ["XAI_API_KEY"]
    # model = "grok-2-latest"

    base_url = "http://147.8.181.248:8000/v1/"
    api_key = "no-api-key"
    model = "/home/mnt/wdxu/models/Qwen2.5-Coder-7B-Instruct"

    with open(os.environ["SWING_DEMO_DATASET_PATH"], "r") as f:
        dataset = json.load(f)
    with open(os.environ["SWING_DEMO_PATCH_PATH"], "r") as f:
        patch = f.read()

    retriever = BM25DiskRetriever(index_dir=os.environ["SWING_INDEXES_PATH"])

    code_editor = RawDataCodeEditor(
        api_key=api_key,
        base_url=base_url,
        model=model
    )
    data = SwingbenchInstance(**dataset[0])
    if not SWING_DEBUG_GENERATE_DRYRUN:
        print('----------- [BEGIN PATCH GENERATOR] -----------',)
        print('input data: ', data)
        patch_generator = PatchGenerator(workdir=os.environ["SWING_TESTBED_PATH"], 
            src_folder=os.environ["SWING_REPOS_DIR_PATH"], 
            code_editor=code_editor,
            retriever=retriever,
            retrieve_file_num=5,
            agent_retry_times=3
        )
        patch = patch_generator.generate(data)
        print('generated patch: ', patch)
        print('----------- [END PATCH GENERATOR] -----------',)

        print('----------- [BEGIN PATCH VERIFIER] -----------')

        patch_verifier = PatchVerifier(ci_tool_name="cargo", 
            workdir=os.environ["SWING_TESTBED_PATH"], 
            src_folder=os.environ["SWING_REPOS_DIR_PATH"], 
        )
        result = patch_verifier.verify(data, patch)
        print('verify result: ', result)
        print('----------- [END PATCH VERIFIER] -----------')

        print('----------- [BEGIN TEST GENERATOR] -----------')
        test_generator = TestGenerator(workdir=os.environ["SWING_TESTBED_PATH"], 
            src_folder=os.environ["SWING_REPOS_DIR_PATH"], 
            code_editor=code_editor,
            retriever=retriever,
            retrieve_file_num=5,
            agent_retry_times=3,
        )
        testcase = test_generator.generate(data, patch)
        print('generated testcase: ', testcase)
        print('----------- [END TEST GENERATOR] -----------')

        print('----------- [BEGIN TEST VERIFIER] -----------')
        test_verifier = TestVerifier(ci_tool_name="cargo", 
            workdir=os.environ["SWING_TESTBED_PATH"], 
            src_folder=os.environ["SWING_REPOS_DIR_PATH"], 
        )
        result = test_verifier.verify(data, testcase)
        print('test verify result: ', result)
        print('----------- [END TEST VERIFIER] -----------')

        print('patch generated instance: ', data)
        print('test generated instance: ', testcase)

        result_patch = merge_diffs(patch, testcase)
        print('patch with test: ', result_patch)

        result = test_verifier.verify(data, result_patch)
        print('patch with test verify result: ', result)

    else:
        import swebench.harness.agent.verifier_test_patch as test_patch
        patch = test_patch.patch
