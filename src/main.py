from langchain_core.messages import HumanMessage
from .retriever import ingest_documents, get_retriever_tools
from .agents.graph import build_graph


def run_question(app, question: str) -> None:
    print(f"\n{'='*60}")
    print(f"❓ Question: {question}")
    print('='*60)
    print("\n🔄 Processing...\n")

    inputs = {"messages": [HumanMessage(content=question)]}
    last_output = None

    for output in app.stream(inputs, {"recursion_limit": 10}):
        for key, value in output.items():
            print(f"📍 Node: {key}")
            if "messages" in value:
                last_msg = value["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print(f"🔧 Tool Call: {last_msg.tool_calls}")
                elif hasattr(last_msg, "content"):
                    print(f"💬 Output:\n{last_msg.content}")
            print()
        last_output = output

    if last_output is None:
        print("⚠️  No output produced.")
        return

    final_message = last_output[list(last_output.keys())[-1]]["messages"][-1]
    print("\n" + "=" * 60)
    print("✨ FINAL ANSWER:")
    print("=" * 60)
    print(final_message.content)
    print("=" * 60)


def main():
    print("🚀 Initializing Agentic RAG System...")

    urls = ["https://techcrunch.com/2026/05/22/how-vcs-and-founders-use-inflated-arr-to-kingmake-ai-startups/"]

    print("\n📚 Ingesting documents into FAISS vector store...")
    try:
        vectorstore = ingest_documents(urls)
        print("✅ Documents ingested successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not ingest documents: {e}")
        import traceback
        traceback.print_exc()
        return

    retriever_tools = get_retriever_tools(vectorstore)
    tools = [retriever_tools]

    print("\n🔧 Building LangGraph state machine...")
    app = build_graph(tools)
    print("✅ Graph built successfully!\n")

    print("=" * 60)
    print("💬 Interactive mode — ask a question, or type 'quit' / 'exit' to leave.")
    print("=" * 60)

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", ":q"}:
            print("👋 Bye!")
            break

        try:
            run_question(app, question)
        except Exception as e:
            print(f"\n⚠️  Error while running question: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
