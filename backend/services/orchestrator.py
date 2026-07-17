import re
from typing import List, Dict, Any, Tuple, Optional
from backend.services.vector_store import VectorStore
from backend.services.llm_service import LLMService

SYSTEM_INSTRUCTION_TEMPLATE = """You are PaperPilot, an advanced research assistant designed to answer queries based on a collection of scientific papers.
You have access to a search tool that allows you to query the papers.

Available papers in the system:
{papers_list}

YOUR TASK:
Answer the user's question accurately using ONLY evidence retrieved from the papers. If the papers do not contain enough information, explain what is missing.

HOW TO USE TOOLS:
To search for relevant information within a specific paper, you MUST use the following XML tag in your response:
<search paper_id="PAPER_ID" query="YOUR_SEARCH_QUERY" />

Rules for tools:
1. You can perform multiple search calls in a single turn if you need information from multiple papers or want to try different search queries.
2. After you output one or more <search /> tags, STOP generating text. The system will run the searches and provide the results in the next turn.
3. You can execute up to 3 rounds of searches.
4. If you already have all the information required, do not output any <search /> tags. Just write your final response.

CITATIONS:
Every claim you make must be grounded and cited. Use the format: [Paper Title, Page X, Section Y] or [Paper Title, Pages X-Y, Section Z].
Only cite sections that were actually returned in the search results.

IMPORTANT: Do not make up information. If a comparison is asked but a paper has no information on the topic, state that clearly.
"""

class Orchestrator:
    def __init__(self, vector_store: VectorStore, llm_service: LLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service

    def query(
        self, 
        question: str, 
        available_papers: List[Dict[str, str]], 
        max_turns: int = 3
    ) -> Dict[str, Any]:
        """
        Runs the agent loop:
        1. Formulates system prompt with available papers list.
        2. Starts chat session with LLM.
        3. Parses <search /> tags, runs query on vector store.
        4. Feeds search results back into chat session.
        5. Returns final response and trace logs of tool calls.
        """
        # Format papers list for system prompt
        papers_list_str = "\n".join([
            f"- ID: {p['id']} | Title: {p['title']}"
            for p in available_papers
        ])
        
        system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(papers_list=papers_list_str)
        
        messages = [{"role": "user", "content": question}]
        trace = []
        final_answer = ""
        all_retrieved_chunks = {}  # Map chunk ID -> chunk dict to avoid duplicates
        
        # Search tag parser regex
        search_regex = re.compile(
            r'<search\s+paper_id=["\'](?P<paper_id>.*?)["\']\s+query=["\'](?P<query>.*?)["\']\s*/?>',
            re.IGNORECASE
        )
        
        for turn in range(max_turns):
            # Generate response from LLM
            response_text = self.llm_service.generate(messages, system_instruction=system_instruction)
            
            # Find all search tags
            search_calls = list(search_regex.finditer(response_text))
            
            if not search_calls:
                # No search tags found; this is the final answer!
                final_answer = response_text
                trace.append({
                    "turn": turn + 1,
                    "action": "respond",
                    "content": response_text
                })
                break
                
            # Log the calls to trace
            calls_list = []
            results_content = []
            
            for match in search_calls:
                paper_id = match.group("paper_id").strip()
                query = match.group("query").strip()
                calls_list.append({"paper_id": paper_id, "query": query})
                
                # Verify paper_id exists in available papers
                paper_exists = any(p["id"] == paper_id for p in available_papers)
                if not paper_exists:
                    results_content.append(
                        f"<results paper_id='{paper_id}' query='{query}'>\n"
                        f"Error: Paper ID '{paper_id}' is not in the system. Available papers: {', '.join([p['id'] for p in available_papers])}.\n"
                        f"</results>"
                    )
                    continue
                
                # Execute search
                # We retrieve up to 5 chunks
                chunks = self.vector_store.search_chunks(query=query, paper_id=paper_id, limit=5)
                
                # Add to accumulated chunks
                for chunk in chunks:
                    all_retrieved_chunks[chunk["id"]] = chunk
                
                # Format chunk results for the LLM
                chunk_lines = []
                for idx, chunk in enumerate(chunks):
                    # Format pages covered
                    pages = ", ".join(map(str, chunk["pages_covered"]))
                    chunk_lines.append(
                        f"Chunk {idx+1} [Page {pages}, Section {chunk['primary_section']}]:\n"
                        f"\"\"\"\n{chunk['text']}\n\"\"\""
                    )
                
                formatted_result = (
                    f"<results paper_id='{paper_id}' query='{query}'>\n"
                    + "\n\n".join(chunk_lines) +
                    f"\n</results>"
                )
                results_content.append(formatted_result)
                
            # Append assistant response and tool execution results to messages
            messages.append({"role": "assistant", "content": response_text})
            
            tool_results_str = "\n\n".join(results_content)
            messages.append({"role": "user", "content": f"Search tool output:\n\n{tool_results_str}"})
            
            trace.append({
                "turn": turn + 1,
                "action": "search",
                "calls": calls_list,
                "response_raw": response_text,
                "results_summary": f"Retrieved {len(search_calls)} search sets"
            })
            
        else:
            # If we exited the loop without breaking (hit max_turns)
            # Request LLM to synthesize final response without calling more tools
            messages.append({
                "role": "user", 
                "content": "You have reached the maximum number of searches. Please synthesize your final answer using the search results you have received so far. Do NOT call <search /> again."
            })
            final_answer = self.llm_service.generate(messages, system_instruction=system_instruction)
            trace.append({
                "turn": max_turns + 1,
                "action": "respond_fallback",
                "content": final_answer
            })
            
        return {
            "answer": final_answer,
            "trace": trace,
            "chunks": list(all_retrieved_chunks.values())
        }
