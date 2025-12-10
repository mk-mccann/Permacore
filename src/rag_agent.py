from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.checkpoint.memory import InMemorySaver  

from utils.citation_formatter import build_citation, format_citation_line


class CustomAgentState(AgentState):  
    user_id: str
    preferences: dict
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[CustomAgentState]):

    state_schema = CustomAgentState

    def __init__(self, vectorstore: Chroma, 
                 k_documents: int = 4, 
                 similarity_threshold: float = 0.25):
        
        self.vectorstore = vectorstore
        self.k_documents = k_documents
        self.similarity_threshold = similarity_threshold

    # This overrides the before_model hook to inject retrieved documents
    def before_model(self, state: CustomAgentState, runtime: Any) -> dict[str, Any]:

        last_message = state["messages"][-1]

        # Handle different content types - extract string for search
        query_text: str
        if isinstance(last_message.content, str):
            query_text = last_message.content

        elif isinstance(last_message.content, list):
            # Extract text from content blocks (common in multimodal messages)
            text_parts = [
                item if isinstance(item, str) else item.get("text", "")
                for item in last_message.content
            ]
            query_text = " ".join(text_parts).strip()

        else:
            # Fallback for unexpected types
            query_text = str(last_message.content)
        
        # Do the similarity search
        retrieved_docs_scores = self.vectorstore.similarity_search_with_score(
            query_text,
            k=self.k_documents
        )

        # Filter docs based on a score threshold if needed
        # By default, LangCain returns cosine distance (lower is better)
        retrieved_docs = [(doc, score) for (doc, score) in retrieved_docs_scores 
                          if 0 < score < self.similarity_threshold]

        # Format context with numbered sources for citation using shared helpers
        docs_content_with_citations = []
        for idx, (doc, score) in enumerate(retrieved_docs, 1):
            citation = build_citation(doc, idx, score)
            docs_content_with_citations.append(
                format_citation_line(citation, include_content=doc.page_content)
            )
        
        docs_content = "\n\n".join(docs_content_with_citations)

        augmented_message_content = (
            f"{last_message.content}\n\n"
            "Use the following context to answer the query. "
            "When using information from the context, cite the source number (e.g., [1]):\n\n"
            f"{docs_content}"
        )
        
        # Provide retrieved docs under "context" (matches state_schema)
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": retrieved_docs,
        }


# class TrimMessagesMiddleware(AgentMiddleware[CustomAgentState]):

#     state_schema = CustomAgentState

#     @before_model
#     def trim_messages(state: CustomAgentState, runtime: Runtime) -> dict[str, Any] | None:
#         """Keep only the last few messages to fit context window."""
#         messages = state["messages"]

#         if len(messages) <= 3:
#             return None  # No changes needed

#         first_msg = messages[0]
#         recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
#         new_messages = [first_msg] + recent_messages

#         return {
#             "messages": [
#                 RemoveMessage(id=REMOVE_ALL_MESSAGES),
#                 *new_messages
#             ]
#         }


#     def delete_specfic_messages(state):
#         messages = state["messages"]
#         if len(messages) > 2:
#             # remove the earliest two messages
#             return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}  
        

#     def delete_all_messages(state):
#         return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}  


class RAGAgent:

    def __init__(self,
        chroma_db_dir: Path | str,
        collection_name: str,
        model_name: str = "mistral-small-latest",
        embeddings_model: str = "mistral-embed",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        k_documents: int = 4,
        similarity_threshold: float = 0.25
    ):
        
        """
        Initialize the RAG agent with conversational memory.
        
        Args:
            chroma_db_dir (Path | str): Directory containing the ChromaDB database.
            collection_name (str): name of the ChromaDB collection.
            model_name (str): Mistral model to use for chat.
            embeddings_model (str): Model to use for embeddings.
            temperature (float): Temperature for response generation.
            max_tokens (int): Maximum tokens in response.
            k_documents (int): Number of documents to retrieve.
            similarity_threshold (float): Similarity score threshold for document filtering.
        """
        
        self.chroma_db_dir = Path(chroma_db_dir)
        self.collection_name= collection_name
        self.k_documents = k_documents
        self.similarity_threshold = similarity_threshold
        
        # Initialize embeddings
        self.embeddings = MistralAIEmbeddings(model=embeddings_model)
        
        # Initialize vectorstore
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(chroma_db_dir)
        )
        
        # Initialize chat model
        self.model = ChatMistralAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Create agent with retrieval middleware
        self.agent = create_agent(
            self.model,
            system_prompt="Please be concise and to the point.",
            tools=[],
            middleware=[RetrieveDocumentsMiddleware(self.vectorstore, 
                                                    self.k_documents,
                                                    self.similarity_threshold)],
            state_schema=CustomAgentState,
            checkpointer=InMemorySaver()
        )

        self.agent.invoke(
            {"messages": [{"role": "user", 
                           "content": "Hi! My source is Bob."
                           }]
                        },
            {"configurable": {"thread_id": "1"}},  
        )


    def chat(self, thread_id: str = "default"):
        """
        Start an interactive chat session.
        
        Args:
            thread_id (str): Unique identifier for this conversation thread.
        """

        print("RAG Agent Chat Interface")
        print("Type 'quit', 'exit', or 'q' to end the conversation")
        print("-" * 50)

        last_context = None    # Initialize last_context to None
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'sources':
                    if last_context:
                        print("\nSources from last response:")
                        for idx, doc in enumerate(last_context):
                            citation = build_citation(doc, idx)
                            print(f"{format_citation_line(citation)}")
                    else:
                        print("No sources available yet. Ask a question first!")
                    continue
                
                if not user_input:
                    continue
                
                # Stream the response
                print("\nAssistant: ", end="", flush=True)
                
                for step in self.agent.stream(
                    {
                        "messages": [{"role": "user", "content": user_input}],
                    },
                    {"configurable": {"thread_id": thread_id}},
                    stream_mode="values"
                ):
                    # Get the last message
                    last_msg = step["messages"][-1]
                    
                    # Only print assistant messages
                    if last_msg.type == "ai":
                        # Print content (this will show incrementally if streaming)
                        print(last_msg.content, end="", flush=True)
                        
                        # Save context for 'sources' command
                        if "context" in step:
                            last_context = step["context"]
                
                print()  # New line after response
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue

    def query(self, question: str, thread_id: str = "default") -> dict:
        """
        Query the agent with a single question.
        
        Args:
            question (str): The question to ask.
            thread_id (str): Thread ID for conversation continuity.
            
        Returns:
            dict: Contains 'answer' and 'sources' keys.
        """
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            {"configurable": {"thread_id": thread_id}}
        )
        
        # Extract answer and sources
        answer = result["messages"][-1].content
        sources = []
        retrieved = result.get("context", []) 

        for idx, doc in enumerate(retrieved):
            citation_dict = build_citation(doc, idx)
            citation = format_citation_line(citation_dict)
            # Expose a consistent, enriched source dict to callers
            sources.append(citation)
        
        return {
            "answer": answer,
            "sources": sources
        }
    

    def _test_query_prompt_with_context(self):
        query = (
            "What is permaculture?"
        )

        for step in self.agent.stream(
            {
                "messages": [{"role": "user", "content": query}],
                "user_id": "user_123",
                "preferences": {"theme": "dark"}
            },
            {"configurable": {"thread_id": "1"}},
            stream_mode="values"
            ):

            step["messages"][-1].pretty_print()


    def _test_query_with_sources(self):
        """Test query that shows citations."""
        query = "What are the principles of permaculture?"
        
        print("Question:", query)
        print("\n" + "="*50 + "\n")
        
        result = self.query(query, thread_id="test")
        
        print("Answer:")
        print(result["answer"])
        print("\n" + "-"*50 + "\n")
        print("Sources:")

        for source in result["context"]:
            print(source)



if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    mistral_api_key = os.getenv("MISTRAL_API_KEY")

    if mistral_api_key:
        mistral_api_key = mistral_api_key.strip()
    else:   
        raise ValueError("MISTRAL_API_KEY not set in environment variables.")

    agent = RAGAgent(
        chroma_db_dir = Path("../chroma_db"),
        collection_name = "ragrarian",
        model_name = "mistral-small-latest",
        embeddings_model = "mistral-embed",
    )
    
    # agent._test_query_prompt_with_context()

    # Option 1: Interactive chat
    # agent.chat(thread_id="session_1")
    
    # Option 2: Single query with sources
    agent._test_query_with_sources()
