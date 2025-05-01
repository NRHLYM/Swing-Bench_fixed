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
                
        elif language == "cpp":
            import tree_sitter_cpp
            CPP_LANGUAGE = Language(tree_sitter_cpp.language())
            self.parser = Parser(CPP_LANGUAGE)
        
            print(f"Successfully loaded C++ language: {self.parser}")
                
        elif language == "go":
            import tree_sitter_go
            GO_LANGUAGE = Language(tree_sitter_go.language())
            self.parser = Parser(GO_LANGUAGE)
        
            print(f"Successfully loaded Go language: {self.parser}")
                
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
        # print("start chunk in here==========================:{}".format(code_snippet))
        
        if not code_snippet or code_snippet.strip() == "":
            return []
            
        # If tree-sitter parser is available, use it
        if self.parser is not None:
                #print("start _chunk_with_tree_sitter in here==========================")
                #assert 1==0
                return self._chunk_with_tree_sitter(code_snippet)

            
        else:
            #assert 1==0
            # Use regex-based chunking as fallback
            return self._chunk_with_regex(code_snippet)
    
    def _chunk_with_tree_sitter(self,  code_snippet: str = None) -> List[Dict[str, Any]]:
        """Use tree-sitter to extract code chunks."""
        #print("start _chunk_with_tree_sitter in here==========================")
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
            #print("self.language:{}".format(self.language))
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


class ChunkScorer:
    """
    代码块评分器，用于将代码块与外部模型集成进行评分
    """
    def __init__(self, model_path=None):
        """
        初始化代码块评分器
        
        Args:
            model_path: 用于评分的模型路径，如果为None则使用默认模型
        """
        self.model_path = model_path
        
        # 导入所需库
        try:
            import json
            import os
            import subprocess
            self.json = json
            self.os = os
            self.subprocess = subprocess
            self.initialized = True
            print(f"ChunkScorer initialized with model: {model_path}")
        except ImportError as e:
            print(f"Error importing dependencies: {e}")
            self.initialized = False
    
    def prepare_scoring_data(self, chunks, problem_statement):
        """
        准备用于评分的数据格式
        
        Args:
            chunks: 代码块列表
            problem_statement: 问题描述
            
        Returns:
            格式化后的数据
        """
        scoring_data = []
        for i, chunk in enumerate(chunks):
            scoring_data.append({
                "id": i,
                "question": problem_statement,
                "code_chunk": chunk["code"],
                "metadata": {
                    "type": chunk.get("type", ""),
                    "name": chunk.get("name", ""),
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0)
                }
            })
        return scoring_data
    
    def save_temp_data(self, data, temp_file="temp_chunks_to_score.json"):
        """保存临时数据到文件"""
        with open(temp_file, "w") as f:
            self.json.dump(data, f)
        return temp_file
    
    def run_scoring_model(self, input_file, output_file, batch_size=4, n_tokens=800):
        """运行外部评分模型"""
        if not self.model_path:
            print("No model path specified, skipping scoring")
            return False
        
        try:
            cmd = [
                "accelerate", "launch",
                "scorer.py",
                f"model_name={self.model_path}",
                f"output_file={output_file}",
                f"batch_size={batch_size}",
                f"dataset_reader.dataset_path={input_file}",
                f"dataset_reader.n_tokens={n_tokens}"
            ]
            
            # 执行打分命令
            result = self.subprocess.run(cmd, check=False, capture_output=True)
            
            if result.returncode != 0:
                print(f"Error running scoring model: {result.stderr.decode()}")
                return False
            
            return True
        except Exception as e:
            print(f"Error during scoring: {e}")
            return False
    
    def load_scored_results(self, output_file):
        """加载评分结果"""
        if not self.os.path.exists(output_file):
            print(f"Scored file not found: {output_file}")
            return None
        
        try:
            with open(output_file, "r") as f:
                return self.json.load(f)
        except Exception as e:
            print(f"Error loading scored results: {e}")
            return None
    
    def score_chunks(self, chunks, problem_statement, output_dir="output", 
                    task_name="code_chunks", batch_size=4, n_tokens=800):
        """
        对代码块进行评分
        
        Args:
            chunks: 代码块列表
            problem_statement: 问题描述
            output_dir: 输出目录
            task_name: 任务名称
            batch_size: 批处理大小
            n_tokens: token数量限制
            
        Returns:
            评分后的代码块列表
        """
        if not self.initialized or not chunks:
            return chunks
        
        # 确保输出目录存在
        self.os.makedirs(output_dir, exist_ok=True)
        
        # 准备评分数据
        scoring_data = self.prepare_scoring_data(chunks, problem_statement)
        
        # 保存临时数据
        temp_file = self.save_temp_data(scoring_data)
        
        # 定义输出文件路径
        output_file = f"{output_dir}/{task_name}_scored.json"
        
        # 运行评分模型
        success = self.run_scoring_model(temp_file, output_file, batch_size, n_tokens)
        
        if not success:
            print("Scoring failed, returning original chunks")
            return chunks
        
        # 加载评分结果
        scored_data = self.load_scored_results(output_file)
        
        if not scored_data:
            return chunks
        
        # 将评分结果合并回原始chunks
        try:
            for item in scored_data:
                chunk_id = item.get("id")
                if 0 <= chunk_id < len(chunks):
                    chunks[chunk_id]["score"] = item.get("score", 0.0)
            
            # 根据分数排序
            sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)
            return sorted_chunks
        except Exception as e:
            print(f"Error processing scored results: {e}")
            return chunks


class ChunkingPipeline:
    """
    代码分块和评分的端到端流水线
    """
    def __init__(self, 
                chunker_language="python", 
                chunker_type="function", 
                reranker_model="microsoft/codebert-base",
                scorer_model_path=None):
        """
        初始化分块评分流水线
        
        Args:
            chunker_language: 代码分块器的语言
            chunker_type: 分块类型 (function, class, block)
            reranker_model: 用于初步重排序的模型名称
            scorer_model_path: 用于最终评分的模型路径
        """
        # 初始化组件
        self.chunker = CodeChunker(language=chunker_language, chunk_type=chunker_type)
        self.reranker = CodeReranker(model_name=reranker_model)
        self.scorer = ChunkScorer(model_path=scorer_model_path)
        
        # 导入所需库
        try:
            import os
            import json
            self.os = os
            self.json = json
            print(f"ChunkingPipeline initialized with {chunker_language}/{chunker_type}")
        except ImportError as e:
            print(f"Error importing dependencies: {e}")
    
    def run(self, 
           code_snippet, 
           problem_statement, 
           output_dir="output", 
           task_name="code_chunks",
           rerank=True,
           top_k=10,
           score=True,
           batch_size=4,
           n_tokens=800):
        """
        运行完整的分块评分流水线
        
        Args:
            code_snippet: 代码片段
            problem_statement: 问题描述
            output_dir: 输出目录
            task_name: 任务名称
            rerank: 是否进行重排序
            top_k: 保留的代码块数量
            score: 是否进行评分
            batch_size: 评分批处理大小
            n_tokens: token数量限制
            
        Returns:
            处理后的代码块列表
        """
        # 确保输出目录存在
        self.os.makedirs(output_dir, exist_ok=True)
        
        # 步骤1: 代码分块
        print(f"Step 1: Chunking code with {self.chunker.language}/{self.chunker.chunk_type}")
        chunks = self.chunker.chunk(code_snippet)
        
        if not chunks:
            print("No chunks were extracted from the code snippet.")
            return []
        
        print(f"Extracted {len(chunks)} chunks from the code")
        
        # 保存原始分块结果
        chunks_file = f"{output_dir}/{task_name}_chunks.json"
        with open(chunks_file, "w") as f:
            self.json.dump([{
                "type": chunk.get("type", ""),
                "name": chunk.get("name", ""),
                "code": chunk["code"],
                "start_line": chunk.get("start_line", 0),
                "end_line": chunk.get("end_line", 0)
            } for chunk in chunks], f, indent=2)
        
        # 步骤2: 重排序（可选）
        if rerank and self.reranker.initialized:
            print(f"Step 2: Reranking chunks, keeping top {top_k}")
            chunks = self.reranker.rerank(chunks, problem_statement, top_k=top_k)
            
            # 保存重排序结果
            reranked_file = f"{output_dir}/{task_name}_reranked.json"
            with open(reranked_file, "w") as f:
                self.json.dump([{
                    "type": chunk.get("type", ""),
                    "name": chunk.get("name", ""),
                    "code": chunk["code"],
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                    "similarity_score": chunk.get("similarity_score", 0.0)
                } for chunk in chunks], f, indent=2)
        else:
            print("Skipping reranking step")
        
        # 步骤3: 评分（可选）
        if score and self.scorer.initialized and self.scorer.model_path:
            print(f"Step 3: Scoring chunks with model {self.scorer.model_path}")
            chunks = self.scorer.score_chunks(
                chunks=chunks,
                problem_statement=problem_statement,
                output_dir=output_dir,
                task_name=task_name,
                batch_size=batch_size,
                n_tokens=n_tokens
            )
            
            # 保存最终评分结果
            final_file = f"{output_dir}/{task_name}_final.json"
            with open(final_file, "w") as f:
                self.json.dump([{
                    "type": chunk.get("type", ""),
                    "name": chunk.get("name", ""),
                    "code": chunk["code"],
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                    "similarity_score": chunk.get("similarity_score", 0.0),
                    "model_score": chunk.get("score", 0.0)
                } for chunk in chunks], f, indent=2)
        else:
            print("Skipping scoring step")
        
        return chunks
    
    def process_file(self, file_path, problem_statement, **kwargs):
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
            problem_statement: 问题描述
            **kwargs: 传递给run方法的其他参数
            
        Returns:
            处理后的代码块列表
        """
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # 设置任务名称为文件名
            task_name = self.os.path.basename(file_path)
            
            # 运行流水线
            return self.run(
                code_snippet=code,
                problem_statement=problem_statement,
                task_name=task_name,
                **kwargs
            )
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []
    
    def process_directory(self, dir_path, problem_statement, 
                        file_extensions=None, recursive=True, **kwargs):
        """
        处理目录中的多个文件
        
        Args:
            dir_path: 目录路径
            problem_statement: 问题描述
            file_extensions: 文件扩展名列表，如['.py', '.js']
            recursive: 是否递归处理子目录
            **kwargs: 传递给run方法的其他参数
            
        Returns:
            {file_path: chunks} 字典
        """
        if file_extensions is None:
            # 根据chunker语言设置默认扩展名
            if self.chunker.language == "python":
                file_extensions = ['.py']
            elif self.chunker.language == "javascript":
                file_extensions = ['.js']
            elif self.chunker.language == "typescript":
                file_extensions = ['.ts', '.tsx']
            elif self.chunker.language == "rust":
                file_extensions = ['.rs']
            else:
                file_extensions = []
        
        results = {}
        
        # 遍历目录
        if recursive:
            for root, _, files in self.os.walk(dir_path):
                for file in files:
                    # 检查文件扩展名
                    if any(file.endswith(ext) for ext in file_extensions):
                        file_path = self.os.path.join(root, file)
                        results[file_path] = self.process_file(
                            file_path=file_path,
                            problem_statement=problem_statement,
                            **kwargs
                        )
        else:
            # 只处理顶级目录
            for file in self.os.listdir(dir_path):
                file_path = self.os.path.join(dir_path, file)
                if self.os.path.isfile(file_path) and any(file.endswith(ext) for ext in file_extensions):
                    results[file_path] = self.process_file(
                        file_path=file_path,
                        problem_statement=problem_statement,
                        **kwargs
                    )
        
        return results
