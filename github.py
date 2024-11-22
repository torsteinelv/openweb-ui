"""
title: Github Analysis Pipeline with Ollama
author: torsteinelv
date: 2024-11-21
version: 1.5
license: MIT
description: A pipeline for analyzing a GitHub repository using Ollama embeddings and a simple in-memory vector store.
requirements:
  - langchain-ollama
  - PyGithub
  - numpy
  - langchain-community
"""

from typing import List, Union, Generator, Iterator
from github import Github, GithubException
import os
import numpy as np
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

class Pipeline:
    class Valves(BaseModel):
        GITHUB_TOKEN: str
        REPO_NAME: str
        OLLAMA_HOST: str
        EMBEDDING_MODEL: str
        LLM_MODEL: str

    def __init__(self):
        self.documents = []  # Store documents in memory
        self.embeddings = []  # Store embeddings in memory
        self.llm = None
        self.qa_chain = None
        self.embedding_model = None

        # Initialize Valves with environment variables or fallback values
        self.valves = self.Valves(
            GITHUB_TOKEN=os.getenv("GITHUB_TOKEN") or "",
            REPO_NAME=os.getenv("REPO_NAME", ""),
            OLLAMA_HOST=os.getenv("OLLAMA_HOST", "http://10.10.0.11:11434"),
            EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            LLM_MODEL=os.getenv("LLM_MODEL", "llama2"),
        )

        # Warn if fallback token is used
        if os.getenv("GITHUB_TOKEN") is None:
            print(
                "WARNING: Using fallback GitHub token. Please set GITHUB_TOKEN as an environment variable."
            )

    async def on_startup(self):
        try:
            print("Initializing Ollama embeddings and LLM...")
            self.embedding_model = OllamaEmbeddings(
                model=self.valves.EMBEDDING_MODEL,
                base_url=self.valves.OLLAMA_HOST,
            )
            self.llm = OllamaLLM(
                model=self.valves.LLM_MODEL,
                base_url=self.valves.OLLAMA_HOST,
            )
            print("Embeddings and LLM initialized successfully.")

            print(f"Accessing GitHub repository: {self.valves.REPO_NAME}...")
            g = Github(self.valves.GITHUB_TOKEN)
            repo = g.get_repo(self.valves.REPO_NAME)

            print("Extracting repository contents...")
            contents = repo.get_contents("")
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    contents.extend(repo.get_contents(file_content.path))
                else:
                    try:
                        content = file_content.decoded_content.decode("utf-8")
                        self.documents.append(
                            {"content": content, "file_path": file_content.path}
                        )
                        print(f"Processed file: {file_content.path}")
                    except Exception as e:
                        print(f"Failed to process file {file_content.path}: {e}")

            print(f"Extracted {len(self.documents)} documents.")

            print("Generating embeddings...")
            for doc in self.documents:
                embedding = self.embedding_model.embed_documents([doc["content"]])[0]
                self.embeddings.append(embedding)

            print("Embeddings generated and stored in memory.")
            print("Pipeline startup complete.")

        except GithubException as e:
            print(f"GitHub API error: {e}")
            raise
        except Exception as e:
            print(f"An error occurred during startup: {e}")
            raise

    async def on_shutdown(self):
        print("Shutting down GitHub Analysis Pipeline...")

    def search_similar(self, query: str, top_k: int = 5):
        query_embedding = self.embedding_model.embed_query(query)
        similarities = [
            (np.dot(query_embedding, doc_embedding), idx)
            for idx, doc_embedding in enumerate(self.embeddings)
        ]
        similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
        return [self.documents[idx]["content"] for _, idx in similarities[:top_k]]

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        if not self.documents or not self.embeddings:
            return "Pipeline not fully initialized. Please check startup process."

        print(f"User Message: {user_message}")
        try:
            relevant_docs = self.search_similar(user_message, top_k=3)
            context = "\n\n".join(relevant_docs)

            query = f"Context: {context}\n\nQuestion: {user_message}\n\nAnswer:"
            response = self.llm.call({"query": query})
            yield response.get("result", "No result found.")
        except Exception as e:
            yield f"An error occurred: {e}"

