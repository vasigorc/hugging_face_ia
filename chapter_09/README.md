# Chapter 9. Programming agents

This chapter introduces the concept of programming agents, focusing on how AI agents can understand, reason, act and respond
to real-world tasks. The author highlights that while specialized models are effective for well-defined tasks, managing
ambiguous, multi step, or dynamic tasks becomes challenging. Agents address this by leveraging Large Language Models (LLMs)
to reason, plan, and delegate tasks, breaking down complex problems into smaller subtasks and using appropriate tools or
models. Agents are able to simplify complex workflows. In this chapter the author sets the stage for building intelligent
agents that can manage state, memory, and tool selection in multi step reasoning pipelines.

## Using LangGraph with tools and memory

The example chosen for this chapter highlights LangGraph, a LangChain extension recommended for building agents with tools
and memory. LangGraph provides flexible graph-based workflow that allows for complex, stateful, and multi step logic. It enables
easy integration of language models, tools, and external APIs, supports branching and dynamic flow control, and is especially
valuable when your application requires persistent memory across multiple steps. This makes it ideal for advanced agents that
need to reason, use tools, and maintain conversational context.
