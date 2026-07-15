from __future__ import annotations

import os
import hashlib
import tempfile
import traceback
from pathlib import Path
from dotenv import load_dotenv

# LangChain LLM adapters
try:
    from langchain_ollama import ChatOllama               # preferred (langchain-ollama package)
except ImportError:
    from langchain_community.chat_models import ChatOllama  # fallback

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)

# Text splitter & embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings          # preferred
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback

# Vector store
from langchain_community.vectorstores import Chroma

# RAG chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# ── Medical-specific system prompt ─────────────────────────────────────────────
MEDICAL_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a literature synthesis assistant for neuro-oncology research, not a clinical decision system.
Use only the supplied context to answer the question. Distinguish source evidence from inference.
If the answer is not supported by the context, say so clearly. Do not provide diagnosis, prognosis, or treatment advice.

Context:
{context}

Question: {question}

Answer (be precise, cite relevant details from the context):"""
)


def find_paper_header_row(frame, search_rows: int = 20) -> int | None:
    """Locate a tracker header row in a formatted Excel sheet."""
    for index, row in frame.head(search_rows).iterrows():
        values = {
            str(value).strip().casefold()
            for value in row.tolist()
            if str(value).strip() and str(value).strip().casefold() != "nan"
        }
        if ({"#", "id"} & values) and ({"paper title", "title"} & values):
            return int(index)
    return None


class RAGService:
    """
    Handles document ingestion into ChromaDB and multi-LLM RAG chat.

    Supported providers:
      - 'medgemma'  → Ollama medgemma:latest  (default, local execution)
      - 'ollama'    → Ollama llama3.2:latest  (local fallback)
      - 'gemini'    → Google Gemini 2.0 Flash (cloud)
      - 'openai'    → OpenAI GPT-4o           (cloud)
    """

    # Ollama model map
    OLLAMA_MODELS = {
        "medgemma": "medgemma:latest",
        "ollama":   "llama3.2:latest",
    }
    INGESTION_SCHEMA_VERSION = "paper_rows_v3_companion_header"

    def __init__(self):
        self.chroma_persist_directory = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "chroma_db"
        )
        os.makedirs(self.chroma_persist_directory, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        self.vectorstore = Chroma(
            persist_directory=self.chroma_persist_directory,
            embedding_function=self.embeddings,
        )
        # Auto-ingest default Literature Tracker if database is empty
        self.auto_ingest_default_tracker()

    # ── LLM Factory ────────────────────────────────────────────────────────────

    def _get_llm(self, provider: str):
        """Return the appropriate LangChain chat model for the given provider."""
        p = provider.lower()

        if p in self.OLLAMA_MODELS:
            model_tag = self.OLLAMA_MODELS[p]
            return ChatOllama(
                model=model_tag,
                temperature=0.1,
                # Pull from Ollama cloud registry automatically if not cached locally
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )

        elif p == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY", "")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set in api/.env")
            return ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                google_api_key=api_key,
                temperature=0.1,
            )

        elif p == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in api/.env")
            return ChatOpenAI(
                model="gpt-4o",
                api_key=api_key,
                temperature=0.1,
            )

        else:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Choose from: medgemma, ollama, gemini, openai"
            )

    # ── Document Ingestion ──────────────────────────────────────────────────────

    def ingest_document(self, file_bytes: bytes, filename: str) -> dict:
        """
        Load, split, embed and store a document into ChromaDB.
        Supports: PDF, TXT, MD, CSV, XLSX.
        """
        file_extension = Path(filename).suffix.lower()
        corpus_version = hashlib.sha256(file_bytes).hexdigest()[:16]

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif file_extension in (".txt", ".md"):
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif file_extension == ".csv":
                loader = CSVLoader(tmp_path)
            elif file_extension in (".xls", ".xlsx"):
                # Preserve one paper/row per Document so retrieved citations can
                # resolve to an exact tracker record rather than a sheet-sized blob.
                import pandas as pd
                from langchain.schema import Document
                xl = pd.ExcelFile(tmp_path)
                documents = []
                for sheet_name in xl.sheet_names:
                    raw = xl.parse(sheet_name, header=None)
                    header_index = find_paper_header_row(raw)
                    if header_index is None:
                        continue
                    df = xl.parse(sheet_name, header=header_index)
                    if df.empty:
                        continue
                    normalized_columns = {
                        str(column).strip().casefold(): str(column)
                        for column in df.columns
                    }
                    id_column = normalized_columns.get("#") or normalized_columns.get("id")
                    title_column = normalized_columns.get("paper title") or normalized_columns.get("title")
                    if not id_column or not title_column:
                        continue
                    for row_offset, (_, row) in enumerate(
                        df.iterrows(), start=header_index + 2
                    ):
                        fields = {}
                        for column, value in row.items():
                            if pd.isna(value) or str(value).strip() == "":
                                continue
                            fields[str(column)] = value.item() if hasattr(value, "item") else value
                        if not fields:
                            continue
                        record_id = str(fields.get(id_column, f"{sheet_name}:{row_offset}"))
                        title = str(fields.get(title_column, ""))
                        if not title.strip():
                            continue
                        doi = str(fields.get(normalized_columns.get("doi", ""), ""))
                        link = str(fields.get(normalized_columns.get("link", ""), ""))
                        year = str(fields.get(normalized_columns.get("year", ""), ""))
                        text = "\n".join(
                            [
                                f"Source sheet: {sheet_name}",
                                f"Record ID: {record_id}",
                                *[f"{key}: {value}" for key, value in fields.items()],
                            ]
                        )
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": filename,
                                    "sheet": sheet_name,
                                    "row": row_offset,
                                    "record_id": record_id,
                                    "title": title,
                                    "doi": doi,
                                    "link": link,
                                    "year": year,
                                    "corpus_version": corpus_version,
                                    "ingestion_schema_version": self.INGESTION_SCHEMA_VERSION,
                                },
                            )
                        )
                loader = None
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            if loader is not None:
                documents = loader.load()
                for document in documents:
                    document.metadata["source"] = filename
                    document.metadata["corpus_version"] = corpus_version
                    document.metadata["ingestion_schema_version"] = self.INGESTION_SCHEMA_VERSION

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=120,
                add_start_index=True,
            )
            chunks = splitter.split_documents(documents)
            chunk_ids = []
            for index, chunk in enumerate(chunks):
                identity = "|".join(
                    [
                        corpus_version,
                        str(chunk.metadata.get("record_id", "")),
                        str(chunk.metadata.get("page", "")),
                        str(chunk.metadata.get("start_index", index)),
                        chunk.page_content,
                    ]
                )
                chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                chunk.metadata["chunk_id"] = chunk_id
                chunk_ids.append(chunk_id)
            try:
                self.vectorstore.delete(ids=chunk_ids)
            except Exception:
                pass
            self.vectorstore.add_documents(documents=chunks, ids=chunk_ids)

            return {
                "status": "success",
                "chunks_ingested": len(chunks),
                "filename": filename,
                "corpus_version": corpus_version,
            }

        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def chat(self, message: str, provider: str = "medgemma", image_base64: str = None) -> dict:
        """
        Run a RAG-powered query against ChromaDB using the specified LLM.
        Defaults to medgemma:latest for local research inference.
        Supports multimodal query if image_base64 is provided.
        Returns a dict containing:
          - answer: The generated answer string
          - sources: A list of source documents with metadata and snippets
        """
        llm = self._get_llm(provider)

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",          # Maximal Marginal Relevance — better diversity
            search_kwargs={"k": 5, "fetch_k": 20},
        )

        if image_base64:
            # For multimodal query, perform manual retrieval first
            source_docs = retriever.invoke(message)
            context_text = "\n\n".join([doc.page_content for doc in source_docs])

            prompt_text = f"""You are a literature synthesis assistant for neuro-oncology research, not a clinical decision system.
Use only the supplied literature context to answer the research question. Treat the attached image as unvalidated research input, not a basis for diagnosis or treatment advice.
If the answer is not supported by the context, say so clearly. Distinguish source evidence from inference.

Context:
{context_text}

Question: {message}

Answer (cite the supporting tracker records and state limitations):"""

            from langchain_core.messages import HumanMessage

            content = [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]

            try:
                msg = HumanMessage(content=content)
                response_val = llm.invoke([msg])
                answer = response_val.content
            except Exception as e:
                # Text-only fallback for local models that don't support multimodal API input
                print(f"[RAGService] Multimodal call failed, falling back to text-only: {e}")
                prompt_fallback = prompt_text + "\n\n(Note: The visual attachment could not be parsed by this model variant. Answer based on text context only.)"
                msg_fallback = HumanMessage(content=prompt_fallback)
                response_val = llm.invoke([msg_fallback])
                answer = response_val.content

            response = {
                "result": answer,
                "source_documents": source_docs
            }
        else:
            # Standard text-only QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": MEDICAL_RAG_PROMPT},
                return_source_documents=True,
            )
            response = qa_chain.invoke({"query": message})

        # Format source documents safely
        sources = []
        seen_source_keys = set()
        for doc in response.get("source_documents", []):
            meta = doc.metadata
            src_name = meta.get("source", "Document")
            # Clean absolute path if it is one
            if "/" in src_name or "\\" in src_name:
                src_name = Path(src_name).name

            page_num = meta.get("page", None)
            if page_num is not None:
                page_num = int(page_num) + 1  # 0-indexed to 1-indexed

            snippet = doc.page_content
            if len(snippet) > 250:
                snippet = snippet[:250].strip() + "..."

            item = {
                "source": src_name,
                "page": page_num,
                "row": meta.get("row"),
                "record_id": meta.get("record_id"),
                "title": meta.get("title"),
                "doi": meta.get("doi"),
                "link": meta.get("link"),
                "corpus_version": meta.get("corpus_version"),
                "chunk_id": meta.get("chunk_id"),
                "snippet": snippet,
            }
            source_key = meta.get("record_id") or meta.get("doi") or meta.get("chunk_id")
            if source_key not in seen_source_keys:
                seen_source_keys.add(source_key)
                sources.append(item)

        return {
            "answer": response.get("result", "No answer generated."),
            "sources": sources
        }

    def get_collection_stats(self) -> dict:
        """Return stats about the current ChromaDB collection."""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            sample = collection.get(limit=min(count, 100), include=["metadatas"]) if count else {"metadatas": []}
            versions = sorted({
                metadata.get("corpus_version")
                for metadata in sample.get("metadatas", [])
                if metadata and metadata.get("corpus_version")
            })
            schema_versions = sorted({
                metadata.get("ingestion_schema_version")
                for metadata in sample.get("metadatas", [])
                if metadata and metadata.get("ingestion_schema_version")
            })
            paper_rows_only = bool(sample.get("metadatas")) and all(
                metadata and metadata.get("record_id") and metadata.get("title")
                for metadata in sample.get("metadatas", [])
            )
            return {
                "document_chunks": count,
                "status": "healthy" if count else "empty",
                "corpus_versions": versions,
                "ingestion_schema_versions": schema_versions,
                "paper_rows_only": paper_rows_only,
                "row_level_provenance": any(
                    metadata and metadata.get("record_id")
                    for metadata in sample.get("metadatas", [])
                ),
            }
        except Exception:
            return {"document_chunks": 0, "status": "empty"}

    def auto_ingest_default_tracker(self) -> None:
        """Check if ChromaDB is empty. If so, automatically ingest default Literature Tracker."""
        try:
            stats = self.get_collection_stats()
            schema_current = stats.get("ingestion_schema_versions") == [self.INGESTION_SCHEMA_VERSION]
            if (
                stats.get("document_chunks", 0) == 0
                or not stats.get("row_level_provenance", False)
                or not stats.get("paper_rows_only", False)
                or not schema_current
            ):
                # Find literature tracker in project root
                root_dir = Path(__file__).resolve().parent.parent.parent
                companion_path = (
                    root_dir
                    / "outputs"
                    / "literature_audit_2026-07-15"
                    / "AI_NeuroOnco_Literature_Audit_and_Index.xlsx"
                )
                source_path = root_dir / "AI_NeuroOnco_Literature_Tracker.xlsx"
                tracker_path = companion_path if companion_path.exists() else source_path
                if tracker_path.exists():
                    if stats.get("document_chunks", 0) > 0:
                        existing = self.vectorstore._collection.get(include=[]).get("ids", [])
                        if existing:
                            self.vectorstore.delete(ids=existing)
                    print(f"[RAGService] Ingesting default Literature Tracker: {tracker_path.name}")
                    with open(tracker_path, "rb") as f:
                        file_bytes = f.read()
                    self.ingest_document(file_bytes, tracker_path.name)
                    print("[RAGService] Default Literature Tracker auto-ingested successfully.")
                else:
                    print(f"[RAGService] Default Literature Tracker not found at {tracker_path}")
        except Exception as e:
            print(f"[RAGService] Failed to auto-ingest default Literature Tracker: {e}")
