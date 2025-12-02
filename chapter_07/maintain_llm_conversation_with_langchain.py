from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

# Use local Ollama model (7B fits comfortably in VRAM for fast inference)
chat_model = ChatOllama(
    model="qwen2.5:7b",
    temperature=1,
)

# Chat prompt template for conversation
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Single session history instance
session_history = InMemoryChatMessageHistory()


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    return session_history  # always return singleton instance in our case


llm_chain = RunnableWithMessageHistory(
    prompt | chat_model | StrOutputParser(),
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

while True:
    user_question = input("Ask a question (type 'exit' to stop): ")

    if user_question.lower() == "quit":
        print("Ending conversation.")
        break

    response = llm_chain.invoke(
        {"question": user_question}, config={"configurable": {"session_id": "default"}}
    )

    print(f"AI: {response}")
