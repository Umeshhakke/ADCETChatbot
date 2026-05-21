import json
import time
from rag_chatbot import answer_query

TEST_FILE = "cutoffqueries.json"


def evaluate():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    total_questions = len(questions)

    print("=" * 100)
    print(f"TOTAL QUESTIONS: {total_questions}")
    print("=" * 100)

    all_results = []

    for index, question in enumerate(questions, start=1):

        print("\n")
        print("=" * 100)
        print(f"QUESTION {index}")
        print("=" * 100)

        print("Q:", question)

        start_time = time.time()

        try:
            answer = answer_query(question)
        except Exception as e:
            answer = f"ERROR: {str(e)}"

        end_time = time.time()

        response_time = round(end_time - start_time, 2)

        print("\nANSWER:")
        print(answer)

        print("\nTIME TAKEN:", response_time, "seconds")

        all_results.append({
            "question_number": index,
            "question": question,
            "answer": answer,
            "response_time_seconds": response_time
        })

    with open("evaluation_cutoff_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n")
    print("=" * 100)
    print("EVALUATION COMPLETED")
    print("=" * 100)
    print("Results saved in evaluation_results.json")


if __name__ == "__main__":
    evaluate()