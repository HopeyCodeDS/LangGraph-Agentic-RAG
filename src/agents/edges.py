from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from ..config import get_llm

class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    reasoning: str = Field(
        description="Brief explanation of why documents are or aren't relevant"
    )

def create_grade_documents():
    llm = get_llm()
    structured_llm = llm.with_structured_output(GradeDocuments)

    def grade_documents(state):
        messages = state["messages"]
        question = messages[0].content

        last_message = messages[-1]

        if not hasattr(last_message, "content"):
            return "generate"
        
        documents_content = last_message.content

        grade_prompt = f"""You are a strict document grader. Your job is to determine if the retrieved documents contain SPECIFIC and DIRECT information to answer the question.

                            Be STRICT in your grading:
                            - Only grade 'yes' if the documents contain clear, specific information that directly answers the question
                            - Grade 'no' if the documents are only tangentially related, too general, or don't contain the specific information needed
                            - Grade 'no' if the documents don't provide enough detail to give a complete answer

                            Question: {question}

                            Retrieved Documents: {documents_content[:500]}...

                            Evaluate: Do these documents contain specific, actionable information to directly answer this question?"""
        
        grade = structured_llm.invoke([SystemMessage(content=grade_prompt)])

        print(f"\n🔍 Grading: {grade.binary_score}")
        print(f"💭 Reasoning: {grade.reasoning}\n")

        if grade.binary_score.lower() == "yes":
            return "generate"
        else:
            return "rewrite"

    return grade_documents


def route_after_agent(state):
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "retrieve"
    else:
        "end"