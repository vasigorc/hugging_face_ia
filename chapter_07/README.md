# Chapter 7. Creating LLM-based applications using LangChain and LlamaIndex

Main takeaways:

- Large Language Models (LLMs) are advanced NLP Models trained on massive
  datasets, capable of understanding and generating humanlike text
- LangChain simplifies building applications with LLMs by chaining components
  like prompts, models and memory
- LlamaIndex allows connecting LLMs with private data via retrieval-augmented
  generation (RAG) for more context-aware responses
- Techniques to maintain conversational context include prompt templates
  with history variables and classes like `RunnableWithMessageHistory`

LLMs contain billions of trainable parameters and processes text by breaking it
into tokens, often at the subword level, enabling understanding and generation
across languages and contexts.

## LangChain

LangChain is a framework designed to streamline creating applications that leverage LLMs.
It organizes component-prompt templates, the LLM actor, memory, and agent chains.

### Using `RunnableWithMessageHistory` class

For the sake of demonstration we will use this book's code example that highlights maintaining
a conversation with the LLM.
