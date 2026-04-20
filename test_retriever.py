# test_retriever.py

from app.services.retriever_service import retrieve_chunks, build_context


def test_one(query: str, file_name: str):
    print(f"\nQuery: {query}")
    print(f"File: {file_name}")
    print("=" * 60)

    try:
        chunks = retrieve_chunks(query=query, file_name=file_name, top_k=3)

        if not chunks:
            print("No chunks were retrieved.")
            return

        print(f"Retrieved {len(chunks)} chunk(s).\n")

        for i, chunk in enumerate(chunks, start=1):
            print(f"--- Chunk {i} ---")
            print(chunk[:800])
            print()

        print("=" * 60)
        print("Combined Context Preview:\n")
        context = build_context(query=query, file_name=file_name, top_k=3)
        print(context[:1500])

    except Exception as e:
        print(f"Error for file '{file_name}': {e}")


def main():
    query = "What is SHAP?"

    candidate_files = [
        "job description_1",
        "job description_2",
        "job description_3",
        "job description_4",
        "job description_5",
        "job description_6",
        "job description_7",
        "CV_Poornima_Jayadevan_7",
        "Cover Letter_7",
    ]

    for file_name in candidate_files:
        test_one(query, file_name)
        print("\n" + "#" * 80 + "\n")


if __name__ == "__main__":
    main()