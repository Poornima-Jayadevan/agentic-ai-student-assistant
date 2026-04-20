# test_tools.py

from app.services.tool_service import calculator_tool, retriever_tool


def main():
    print("=== Calculator Tool ===")
    print(calculator_tool("25 * 3"))
    print(calculator_tool("(10 + 5) / 3"))
    print(calculator_tool("10 / 0"))

    print("\n=== Retriever Tool ===")
    result = retriever_tool(
        query="What skills are required for this role?",
        file_name="job description_1",
        top_k=2,
    )
    print(result[:1500])


if __name__ == "__main__":
    main()