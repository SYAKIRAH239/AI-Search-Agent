from dotenv import load_dotenv
import time
from typing import Annotated, List
from openai import RateLimitError
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from web_operations import serp_search, reddit_search_api, reddit_post_retrieval
from prompts import (
    get_reddit_analysis_messages,
    get_google_analysis_messages,
    get_bing_analysis_messages,
    get_reddit_url_analysis_messages,
    get_synthesis_messages
)

load_dotenv()

llm = init_chat_model("gpt-4o")


def invoke_with_retry(target_llm, messages, max_retries=3, base_delay=5):
    """
    Calls target_llm.invoke(messages), retrying with increasing delay if OpenAI
    returns a rate limit error (429 / tokens-per-minute). Raises if still failing
    after max_retries.
    """
    for attempt in range(max_retries):
        try:
            return target_llm.invoke(messages)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                print(f"Still rate limited after {max_retries} attempts, giving up.")
                raise
            wait = base_delay * (attempt + 1)
            print(f"Rate limited, waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
            time.sleep(wait)

#create state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_question: str | None
    google_results: str | None
    bing_results: str | None
    reddit_results: str | None
    selected_reddit_urls: list[str] | None
    reddit_post_data: list | None
    google_analysis: str | None
    bing_analysis: str | None
    reddit_analysis: str | None
    final_answer: str | None
    needs_research: bool | None


class RedditURLAnalysis(BaseModel):
    selected_urls: List[str] = Field(description="List of Reddit URLs that contain valuable information for answering the user's question")


class RedditRelevanceFilter(BaseModel):
    relevant_indices: List[int] = Field(description="Indices (0-based) of posts that are actually relevant to the user's question, excluding off-topic results")


class RouteDecision(BaseModel):
    needs_research: bool = Field(description="True if answering this question requires new web/reddit research, False if it can be answered using the research already gathered earlier in this conversation")


def route_question(state: State):
    has_prior_research = bool(state.get("final_answer"))

    if not has_prior_research:
        print("No prior research in this conversation, running full research pipeline")
        return {"needs_research": True}

    messages = state.get("messages", [])
    google_analysis = state.get("google_analysis", "")
    bing_analysis = state.get("bing_analysis", "")
    reddit_analysis = state.get("reddit_analysis", "")

    routing_prompt = [
        {
            "role": "system",
            "content": (
                "You are deciding whether the user's latest message requires fresh web/reddit "
                "research, or whether it can be answered using research already gathered earlier "
                "in this conversation.\n\n"
                "Existing research summary:\n"
                f"Google: {google_analysis}\n\nBing: {bing_analysis}\n\nReddit: {reddit_analysis}\n\n"
                "If the latest message is a follow-up, clarification, or asks for more detail about "
                "the same topic already researched, return needs_research=False. "
                "If it introduces a new topic, asks about something not covered above, or explicitly "
                "asks for updated/current information, return needs_research=True."
            ),
        },
        *messages[-6:],  # last few turns for context, avoids sending the whole history every time
    ]

    structured_llm = llm.with_structured_output(RouteDecision)
    try:
        decision = invoke_with_retry(structured_llm, routing_prompt)
        print(f"Routing decision: needs_research={decision.needs_research}")
        return {"needs_research": decision.needs_research}
    except Exception as e:
        print(f"Routing failed, defaulting to running fresh research: {e}")
        return {"needs_research": True}


def route_after_classification(state: State):
    if state.get("needs_research"):
        return ["google_search", "bing_search", "reddit_search"]
    return ["answer_from_context"]


def answer_from_context(state: State):
    print("Answering from existing research (no new search needed)")

    messages = state.get("messages", [])
    google_analysis = state.get("google_analysis", "")
    bing_analysis = state.get("bing_analysis", "")
    reddit_analysis = state.get("reddit_analysis", "")

    context_prompt = [
        {
            "role": "system",
            "content": (
                "You already researched this topic earlier in the conversation. Use the research "
                "below, plus the conversation history, to answer the user's latest message. "
                "If something genuinely isn't covered by this research, say so honestly rather than "
                "making it up.\n\n"
                f"Google findings: {google_analysis}\n\n"
                f"Bing findings: {bing_analysis}\n\n"
                f"Reddit findings: {reddit_analysis}"
            ),
        },
        *messages,
    ]

    try:
        reply = invoke_with_retry(llm, context_prompt)
        answer = reply.content
    except Exception as e:
        print(f"Follow-up answer failed: {e}")
        answer = "Sorry, I ran into an error trying to answer that. Could you try rephrasing?"

    return {"final_answer": answer, "messages": [{"role": "assistant", "content": answer}]}


def google_search(state: State):
    user_question = state.get("user_question", "")
    print(f"Searching Google for: {user_question}")

    google_results = serp_search(user_question, engine="google")

    return {"google_results": google_results}


def bing_search(state: State):
    user_question = state.get("user_question", "")
    print(f"Searching Bing for: {user_question}")

    bing_results = serp_search(user_question, engine="bing")

    return {"bing_results": bing_results}


def reddit_search(state: State):
    user_question = state.get("user_question", "")
    print(f"Searching Reddit for: {user_question}")

    reddit_results = reddit_search_api(keyword=user_question)
    print(reddit_results)

    return {"reddit_results": reddit_results}


def filter_reddit_relevance(state: State):
    print("Filtering reddit results for relevance")

    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")

    posts = reddit_results.get("parsed_posts", []) if reddit_results else []
    posts = [p for p in posts if p.get("title") and p.get("url")]

    if not posts:
        print("No reddit posts available to filter (search may have timed out or returned nothing)")
        return {"reddit_results": None}

    titles_list = "\n".join(f"{i}: {p['title']}" for i, p in enumerate(posts))

    structured_llm = llm.with_structured_output(RedditRelevanceFilter)
    messages = [
        {
            "role": "system",
            "content": (
                "Given a user's question and a list of Reddit post titles, return the indices "
                "of posts that are actually relevant to answering the question. Be strict — "
                "exclude tangentially related, off-topic, or unrelated posts even if they share "
                "a keyword with the question."
            ),
        },
        {"role": "user", "content": f"Question: {user_question}\n\nPosts:\n{titles_list}"},
    ]

    try:
        result = invoke_with_retry(structured_llm, messages)
        relevant_posts = [posts[i] for i in result.relevant_indices if 0 <= i < len(posts)]
        print(f"Kept {len(relevant_posts)}/{len(posts)} reddit posts as relevant")
    except Exception as e:
        print(f"Relevance filtering failed, keeping all posts: {e}")
        relevant_posts = posts

    return {"reddit_results": {"parsed_posts": relevant_posts, "total_found": len(relevant_posts)}}


def analyze_reddit_posts(state: State):
    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")

    if not reddit_results or not reddit_results.get("parsed_posts"):
        print("No reddit posts to select URLs from, skipping")
        return {"selected_reddit_urls": []}

    structured_llm = llm.with_structured_output(RedditURLAnalysis)
    messages = get_reddit_url_analysis_messages(user_question, reddit_results)

    try:
        analysis = invoke_with_retry(structured_llm, messages)
        selected_urls = analysis.selected_urls

        print("Selected URLs:")
        for i, url in enumerate(selected_urls, 1):
            print(f"   {i}. {url}")

    except Exception as e:
        print(e)
        selected_urls = []

    return {"selected_reddit_urls": selected_urls}


def retrieve_reddit_posts(state: State):
    print("Getting reddit post comments")

    selected_urls = state.get("selected_reddit_urls", [])

    if not selected_urls:
        return {"reddit_post_data": []}

    print(f"Processing {len(selected_urls)} Reddit URLs")

    reddit_post_data = reddit_post_retrieval(selected_urls)

    if reddit_post_data:
        print(f"Successfully got {len(reddit_post_data)} posts")
    else:
        print("Failed to get post data")
        reddit_post_data = []

    print(reddit_post_data)
    return {"reddit_post_data": reddit_post_data}


def analyze_google_results(state: State):
    print("Analyzing google search results")

    user_question = state.get("user_question", "")
    google_results = state.get("google_results", "")

    messages = get_google_analysis_messages(user_question, google_results)

    try:
        reply = invoke_with_retry(llm, messages)
        content = reply.content
    except Exception as e:
        print(f"Google analysis failed: {e}")
        content = "Google analysis unavailable due to an error."

    return {"google_analysis": content}


def analyze_bing_results(state: State):
    print("Analyzing bing search results")

    user_question = state.get("user_question", "")
    bing_results = state.get("bing_results", "")

    messages = get_bing_analysis_messages(user_question, bing_results)

    try:
        reply = invoke_with_retry(llm, messages)
        content = reply.content
    except Exception as e:
        print(f"Bing analysis failed: {e}")
        content = "Bing analysis unavailable due to an error."

    return {"bing_analysis": content}


def analyze_reddit_results(state: State):
    print("Analyzing reddit search results")

    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")
    reddit_post_data = state.get("reddit_post_data", "")

    messages = get_reddit_analysis_messages(user_question, reddit_results, reddit_post_data)

    try:
        reply = invoke_with_retry(llm, messages)
        content = reply.content
    except Exception as e:
        print(f"Reddit analysis failed: {e}")
        content = "Reddit analysis unavailable due to an error."

    return {"reddit_analysis": content}


def synthesize_analyses(state: State):
    print("Combine all results together")

    user_question = state.get("user_question", "")
    google_analysis = state.get("google_analysis", "")
    bing_analysis = state.get("bing_analysis", "")
    reddit_analysis = state.get("reddit_analysis", "")

    messages = get_synthesis_messages(
        user_question, google_analysis, bing_analysis, reddit_analysis
    )

    try:
        reply = invoke_with_retry(llm, messages)
        final_answer = reply.content
    except Exception as e:
        print(f"Synthesis failed: {e}")
        final_answer = "Sorry, I couldn't generate a final answer due to a repeated API error. Here's what was found:\n\n" \
                        f"Google: {google_analysis}\n\nBing: {bing_analysis}\n\nReddit: {reddit_analysis}"

    return {"final_answer": final_answer, "messages": [{"role": "assistant", "content": final_answer}]}


graph_builder = StateGraph(State)

#define node/edges
graph_builder.add_node("route_question", route_question)
graph_builder.add_node("answer_from_context", answer_from_context)
graph_builder.add_node("google_search", google_search)
graph_builder.add_node("bing_search", bing_search)
graph_builder.add_node("reddit_search", reddit_search)
graph_builder.add_node("filter_reddit_relevance", filter_reddit_relevance)
graph_builder.add_node("analyze_reddit_posts", analyze_reddit_posts)
graph_builder.add_node("retrieve_reddit_posts", retrieve_reddit_posts)
graph_builder.add_node("analyze_google_results", analyze_google_results)
graph_builder.add_node("analyze_bing_results", analyze_bing_results)
graph_builder.add_node("analyze_reddit_results", analyze_reddit_results)
graph_builder.add_node("synthesize_analyses", synthesize_analyses)

graph_builder.add_edge(START, "route_question")
graph_builder.add_conditional_edges("route_question", route_after_classification)

graph_builder.add_edge("google_search", "analyze_reddit_posts")
graph_builder.add_edge("bing_search", "analyze_reddit_posts")

graph_builder.add_edge("reddit_search", "filter_reddit_relevance")
graph_builder.add_edge("filter_reddit_relevance", "analyze_reddit_posts")

graph_builder.add_edge("analyze_reddit_posts", "retrieve_reddit_posts")

graph_builder.add_edge("retrieve_reddit_posts", "analyze_google_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyze_bing_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyze_reddit_results")

graph_builder.add_edge("analyze_google_results", "synthesize_analyses")
graph_builder.add_edge("analyze_bing_results", "synthesize_analyses")
graph_builder.add_edge("analyze_reddit_results", "synthesize_analyses")

graph_builder.add_edge("synthesize_analyses", END)
graph_builder.add_edge("answer_from_context", END)

graph = graph_builder.compile()


def run_chatbot():
    print("Multi-Source Research Agent")
    print("Type 'exit' to quit\n")

    state = {
        "messages": [],
        "user_question": None,
        "google_results": None,
        "bing_results": None,
        "reddit_results": None,
        "selected_reddit_urls": None,
        "reddit_post_data": None,
        "google_analysis": None,
        "bing_analysis": None,
        "reddit_analysis": None,
        "final_answer": None,
        "needs_research": None,
    }

    while True:
        user_input = input("Ask me anything: ")
        if user_input.lower() == "exit":
            print("Bye")
            break

        state["user_question"] = user_input
        state["messages"] = state["messages"] + [{"role": "user", "content": user_input}]

        print("\nThinking...\n")
        final_state = graph.invoke(state)
        state = final_state  # carry everything forward across turns, including message history

        if final_state.get("final_answer"):
            print(f"\nFinal Answer:\n{final_state.get('final_answer')}\n")

        print("-" * 80)


if __name__ == "__main__":
    run_chatbot()
