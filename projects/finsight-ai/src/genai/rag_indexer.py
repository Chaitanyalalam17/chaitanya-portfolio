
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"

COLLECTION_NAME = "finsight_knowledge"


def load_documents():
    documents = []

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge folder not found: {KNOWLEDGE_DIR}"
        )

    for file_path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")

        sections = text.split("\n## ")

        for index, section in enumerate(sections):
            section = section.strip()

            if not section:
                continue

            if index > 0:
                section = "## " + section

            # Skip sections containing only a heading
            lines = section.splitlines()

            if (
                len(lines) == 1
                and lines[0].strip().startswith("#")
            ):
                continue

            documents.append(
                {
                    "id": f"{file_path.stem}_{index}",
                    "text": section,
                    "source": file_path.name,
                }
            )

    return documents


def build_index():
    documents = load_documents()

    if not documents:
        raise ValueError(
            "No valid Markdown knowledge sections found."
        )

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    embedding_function = DefaultEmbeddingFunction()

    # Rebuild the collection to remove old sections.
    # This affects only the FinSight knowledge collection.
    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
    except chromadb.errors.NotFoundError:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    collection.add(
        ids=[
            doc["id"]
            for doc in documents
        ],
        documents=[
            doc["text"]
            for doc in documents
        ],
        metadatas=[
            {"source": doc["source"]}
            for doc in documents
        ],
    )

    print("\nFinSight AI Knowledge Index")
    print("-" * 40)
    print(f"Documents indexed: {len(documents)}")
    print(f"Total stored sections: {collection.count()}")
    print(f"Vector database: {VECTOR_DB_DIR}")


if __name__ == "__main__":
    build_index()
