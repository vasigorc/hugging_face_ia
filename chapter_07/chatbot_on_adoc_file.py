import argparse
import hashlib
from pathlib import Path

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.docling import DoclingReader
import gradio as gr

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--file",
    help="path to Asciidoc document that will be used for context",
    type=str,
    required=True,
)
parser.add_argument(
    "--force-reload",
    help="force re-indexing even if index already exists",
    action="store_true",
)
parser.add_argument(
    "-t",
    "--temperature",
    help="temperature of the LLM model. 0 - most factual, 2 - most random. Default 0.1",
    type=float,
    default=0.1,
)
args = parser.parse_args()
if not 0 <= args.temperature <= 2:
    parser.error("temperature must be between 0 and 2")

# Configure embedding model globally
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Resolve the file path
file_path = Path(args.file).expanduser().resolve()

# Create index directory name based on file name hash (to handle long paths)
index_dir_name = (
    f".index_{file_path.stem}_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}"
)
index_dir = Path(__file__).parent / index_dir_name

if index_dir.exists() and not args.force_reload:
    # Load existing index
    print(f"Loading existing index from {index_dir}")
    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    index = load_index_from_storage(storage_context)
else:
    # Create new index
    print(f"Creating new index for {file_path}")
    reader = DoclingReader()
    document = reader.load_data(file_path)
    index = VectorStoreIndex.from_documents(document)
    index.storage_context.persist(persist_dir=str(index_dir))
    print(f"Index saved to {index_dir}")

# use a LLM for querying
chat_model = Ollama(model="qwen2.5:7b", temperature=args.temperature)

query_engine = index.as_query_engine(llm=chat_model)


# attach a web frontend for the app
def my_chatbot(input_text):
    response = query_engine.query(input_text)
    return str(response)


with gr.Blocks() as mychatbot:
    chatbot = gr.Chatbot()
    question = gr.Textbox()

    def chat(message, chat_history):
        content = my_chatbot(message)
        chat_history.append((message, content))
        return "", chat_history

    question.submit(fn=chat, inputs=[question, chatbot], outputs=[question, chatbot])

mychatbot.launch()
