"""
title: Smart GitHub Analysis Pipeline with Ollama
author: torsteinelv
date: 2024-11-21
version: 2.2
license: MIT
description: An enhanced pipeline for analyzing a GitHub repository using Ollama embeddings, with contextual memory and improved vector store integration.
requirements:
  - langchain-ollama
  - PyGithub
  - numpy
  - langchain-community
  - pydantic
"""

from typing import List, Union, Generator, Iterator
import os
import numpy as np
from github import Github, GithubException
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import logging

class Pipeline:
    class Valves(BaseModel):
        GITHUB_TOKEN: str
        REPO_NAME: str
        OLLAMA_HOST: str
        EMBEDDING_MODEL: str
        LLM_MODEL: str

    def __init__(self):
        self.documents = []  # Store documents in memory, consider using a database for larger repositories to prevent memory overflow
        self.embeddings = []  # Store embeddings in memory
        self.llm = None
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
            logging.warning("Using fallback GitHub token. Please set GITHUB_TOKEN as an environment variable.")

    async def on_startup(self):
        try:
            logging.info("Initializing Ollama embeddings and LLM...")
            self.embedding_model = OllamaEmbeddings(
                model=self.valves.EMBEDDING_MODEL,
                base_url=self.valves.OLLAMA_HOST,
            )
            self.llm = OllamaLLM(
                model=self.valves.LLM_MODEL,
                base_url=self.valves.OLLAMA_HOST,
            )
            logging.info("Embeddings and LLM initialized successfully.")

            # Access GitHub repository and extract content asynchronously
            logging.info(f"Accessing GitHub repository: {self.valves.REPO_NAME}...")
            g = Github(self.valves.GITHUB_TOKEN)
            repo = g.get_repo(self.valves.REPO_NAME)
            await self.extract_repository_contents(repo)

            logging.info("Generating embeddings...")
            await self.generate_embeddings()

            logging.info("Embeddings generated and stored in memory.")
            logging.info("Pipeline startup complete.")

        except GithubException as e:
            logging.error(f"GitHub API error: {e}")  # Use logging framework for better error tracking
            raise
        except Exception as e:
            logging.error(f"An error occurred during startup: {e}")
            raise

    async def extract_repository_contents(self, repo):
        try:
            logging.info("Extracting repository contents...")
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
                        logging.info(f"Processed file: {file_content.path}")
                    except Exception as e:
                        logging.warning(f"Failed to process file {file_content.path}: {e}")
            logging.info(f"Extracted {len(self.documents)} documents.")
        except Exception as e:
            logging.error(f"Error extracting repository contents: {e}")

    async def generate_embeddings(self):
        try:
            for doc in self.documents:
                embedding = self.embedding_model.embed_documents([doc["content"]])[0]
                self.embeddings.append(embedding)
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")

    async def on_shutdown(self):
        logging.info("Shutting down GitHub Analysis Pipeline...")

    def search_similar(self, query: str, top_k: int = 5):
        try:
            query_embedding = self.embedding_model.embed_query(query)
            similarities = [
                (np.dot(query_embedding, doc_embedding), idx)
                for idx, doc_embedding in enumerate(self.embeddings)
            ]
            similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
            return [
                {
                    "file_path": self.documents[idx]["file_path"],
                    "content": self.documents[idx]["content"]  # Return the full content of the file
                }
                for _, idx in similarities[:top_k]
            ]
        except Exception as e:
            logging.error(f"An error occurred during search: {e}")
            return []

    def find_files_containing(self, text: str) -> List[str]:
        """Find all files containing the specified text."""
        matching_files = [
            doc["file_path"] for doc in self.documents if text in doc["content"]
        ]
        return matching_files

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        if not self.documents or not self.embeddings:
            return "Pipeline not fully initialized. Please check startup process."

        logging.info(f"User Message: {user_message}")
        try:
            relevant_docs = self.search_similar(user_message, top_k=3)
            context = "\n\n".join(
                [f"File: {doc['file_path']}\nContent:\n```\n{doc['content']}\n```" for doc in relevant_docs]
            )

            query = f"Context: {context}\n\nQuestion: {user_message}\n\nAnswer:"
            response = self.llm(query)
            yield response
        except Exception as e:
            yield f"An error occurred: {e}"
