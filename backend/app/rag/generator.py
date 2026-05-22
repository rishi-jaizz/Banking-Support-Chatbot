"""
LLM response generation module.
Uses Google Gemini, Groq, or OpenAI to generate context-aware responses
based on retrieved documents. Includes follow-up question suggestions,
retry handling with backoff, conversational bypass, metadata topic summaries,
and robust static fallbacks.
"""

import logging
import httpx
import re
import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator
from google import genai
from app.config import settings
from app.rag.retriever import retriever
from app.models.schemas import SourceChunk

logger = logging.getLogger(__name__)


# Production-grade system prompts with anti-hallucination guardrails

# 1. Standard RAG Prompt
SYSTEM_PROMPT = """You are BankAssist AI, an expert Banking Support Assistant. You help customers with banking-related queries using ONLY the retrieved context provided to you.

## STRICT RULES — Follow these at all times:
1. **ONLY use information from the "Retrieved Context" section below.** Never fabricate rates, fees, policies, or numbers.
2. If the context does NOT contain enough information to fully answer the question, explicitly state: "Based on the available information, I don't have complete details on this. Please contact your bank directly for accurate information."
3. NEVER invent bank names, branch details, account numbers, or specific dates not in the context.
4. When quoting specific numbers (interest rates, fees, limits), always add a disclaimer like "(as per our knowledge base — please verify with your bank for current rates)".

## RESPONSE FORMAT GUIDELINES:
- Use **bold** for key terms and figures.
- Use bullet points for lists of features, requirements, or steps.
- Use ### headings for different sections when the answer is long.
- Keep responses concise but thorough — aim for 100-300 words.
- End with a brief helpful closing line.

## CITATION RULES:
- When referencing specific information, naturally mention the source topic (e.g., "According to our loans documentation..." or "As outlined in our credit card guidelines...").
- Do NOT make up citations or reference documents not in the context.

## BOUNDARIES:
- You are NOT authorized to perform any banking transactions.
- Do NOT ask for or store personal/financial information (account numbers, PINs, passwords).
- If asked about non-banking topics, politely redirect: "I'm specialized in banking support. For this query, I'd recommend consulting the appropriate service."
- Always recommend customers verify details with their specific bank branch.

## FOLLOW-UP SUGGESTIONS:
After your main answer, suggest 2-3 relevant follow-up questions the customer might want to ask. Format them as:
**💡 You might also want to ask:**
- [suggestion 1]
- [suggestion 2]
- [suggestion 3]"""


# 2. Conversational System Prompt for Greetings
CONVERSATIONAL_SYSTEM_PROMPT = """You are BankAssist AI, an expert Banking Support Assistant.
Provide a friendly, welcoming, and professional conversational response.
Explain that you can help with questions about banking products and services covered in the knowledge base, such as:
- **Savings Accounts** (features, interest rates, balance requirements)
- **Home & Personal Loans** (eligibility, documentation, processes)
- **Credit Cards** (fees, spend limits, reward schemes)
- **UPI & Digital Banking** (NEFT/RTGS/IMPS transfer modes and transaction limits)
- **Fixed Deposits** (rates, senior citizen benefits)
- **Insurance** (policies, claims)
- **KYC & General Policies** (video KYC, document lists)

Keep the response brief, engaging, and invite the user to ask a question or upload documents. Do not search the database for this.
Format 2-3 follow-up suggestions at the end, for example:
**💡 You might also want to ask:**
- What topics are covered in the knowledge base?
- How do I apply for a home loan?
- What are the credit card annual fees?"""


# 3. Metadata System Prompt for topics/capabilities
METADATA_SYSTEM_PROMPT = """You are BankAssist AI, an expert Banking Support Assistant.
You are asked about the topics covered in the knowledge base or the documents that have been uploaded.
Use the list of uploaded documents and topics provided to you to describe what is currently available in the knowledge base.
Be professional, structured (use bullet points), and explain what categories of questions you can answer based on these files.
Explain that users can also upload new documents to expand this knowledge base.
Format 2-3 follow-up suggestions at the end, for example:
**💡 You might also want to ask:**
- What documents are needed for KYC?
- Tell me about fixed deposits
- What is the limit for UPI transactions?"""


# 4. Fallback System Prompt for out-of-domain / weak context queries
FALLBACK_SYSTEM_PROMPT = """You are BankAssist AI, an expert Banking Support Assistant.
The customer has asked a question, but there is no direct or relevant information in the knowledge base to answer it.
Politely inform the customer that you don't have this specific information in your knowledge base.
Suggest the closest available banking topics they can ask about instead, from:
- Savings Accounts (interest rates, types, requirements)
- Loans (home, personal, eligibility)
- Credit Cards (annual fees, features, charges)
- Fixed Deposits & Insurance
- Digital Banking, UPI limits, NEFT/RTGS/IMPS transfers
- KYC requirements

Do NOT invent answers or details not in the knowledge base. Keep the tone helpful and professional.
Format 2-3 follow-up suggestions at the end, for example:
**💡 You might also want to ask:**
- What topics are covered in the knowledge base?
- How do I apply for a home loan?
- What credit cards are available?"""


# Static response Fallbacks if LLM APIs are completely down
STATIC_GREETING_RESPONSE = (
    "Hello! I am **BankAssist AI**, your virtual banking support assistant. "
    "I can help you with questions regarding our banking products and services.\n\n"
    "Here are the topics currently covered in my knowledge base:\n"
    "- **Savings Accounts**: Types of accounts, interest rates, and requirements.\n"
    "- **Home & Personal Loans**: Eligibility, documentation, and application steps.\n"
    "- **Credit Cards**: Features, annual fees, and charge structures.\n"
    "- **Digital Banking & UPI**: Online fund transfers (NEFT/RTGS/IMPS) and transaction limits.\n"
    "- **KYC & General Policies**: Document requirements and verification procedures.\n"
    "- **Fixed Deposits**: Interest rates, tenures, and senior citizen benefits.\n"
    "- **Insurance Policies**: Basic coverage, policies, and claims.\n\n"
    "How can I assist you today?"
)
STATIC_GREETING_SUGGESTIONS = [
    "What topics are covered in the knowledge base?",
    "How can I apply for a home loan?",
    "What credit cards are available?"
]

STATIC_META_SUGGESTIONS = [
    "What documents are needed for KYC?",
    "Tell me about fixed deposits",
    "What is the limit for UPI transactions?"
]

STATIC_FALLBACK_RESPONSE = (
    "I apologize, but I couldn't find any relevant information in our knowledge base to answer your question. "
    "I can only assist with loans, credit cards, fixed deposits, and general banking policies documented in my knowledge base. "
    "Please try rephrasing your question or check with your bank branch for assistance."
)
STATIC_FALLBACK_SUGGESTIONS = [
    "What topics are covered in the knowledge base?",
    "How can I apply for a home loan?",
    "What credit cards are available?"
]


def extract_suggestions(text: str) -> tuple[str, list[str]]:
    """
    Extract follow-up questions from the LLM response text.
    Strips the suggested questions section from the text and returns a list of questions.
    """
    pattern = r"(?:\*\*|\*|)?💡\s*(?:You might also want to ask|Suggested Questions):?(?:\*\*|\*|)?\s*\n?"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    if len(parts) > 1:
        main_text = parts[0].strip()
        suggestions_text = parts[1]
        suggestions = []
        for line in suggestions_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove leading bullet points or numbers (e.g. -, *, 1., 2.)
            line = re.sub(r"^[-*•\d\.\s]+", "", line).strip()
            # Remove trailing/leading quotes if any
            line = line.strip("\"'")
            if line:
                suggestions.append(line)
        return main_text, suggestions
    return text, []


def is_greeting_or_chat(query: str) -> bool:
    """Classify if query is a simple greeting or casual chat."""
    q_clean = query.strip().lower().strip("?.!,")
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings", "yo", "howdy", "hola", "sup"}
    casual = {"thanks", "thank you", "ok", "okay", "cool", "great", "awesome", "bye", "goodbye"}
    if q_clean in greetings or q_clean in casual:
        return True
    
    # Check for short queries that don't have banking terms
    words = q_clean.split()
    banking_terms = {"loan", "card", "savings", "fd", "deposit", "rate", "fee", "upi", "kyc", "transfer", "neft", "rtgs", "imps", "limit", "document"}
    if len(words) <= 2 and not any(w in banking_terms for w in words):
        return True
    return False


def is_meta_query(query: str) -> bool:
    """Classify if query is requesting capability or topic list information."""
    q_clean = query.strip().lower().strip("?.!,")
    meta_keywords = ["topic", "covered", "document", "knowledge base", "what can you do", "help", "capability", "menu", "support", "features", "uploaded"]
    return any(kw in q_clean for kw in meta_keywords)


class ResponseGenerator:
    """
    Generates context-aware responses using Google Gemini, Groq, or OpenAI LLMs.
    Combines retrieved context with conversation history for coherent responses.
    Includes retry loops with exponential backoff and dynamic fallbacks.
    """

    def __init__(self):
        self.client = None
        self.provider = "gemini"
        self.model_name = settings.LLM_MODEL
        self._initialized = False

    def initialize(self):
        """Initialize the client based on LLM provider."""
        self.provider = settings.LLM_PROVIDER.lower()
        self.model_name = settings.LLM_MODEL

        if self.provider == "gemini":
            if not settings.GOOGLE_API_KEY:
                logger.error("GOOGLE_API_KEY not set!")
                raise ValueError("GOOGLE_API_KEY environment variable is required")
            self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
            logger.info(f"Gemini client initialized with model: {self.model_name}")
        elif self.provider == "groq":
            if not settings.GROQ_API_KEY:
                logger.error("GROQ_API_KEY not set!")
                raise ValueError("GROQ_API_KEY environment variable is required for Groq")
            if "gemini" in self.model_name.lower():
                self.model_name = "llama-3.3-70b-versatile"
            logger.info(f"Groq provider selected with model: {self.model_name}")
        elif self.provider == "openai":
            if not settings.OPENAI_API_KEY:
                logger.error("OPENAI_API_KEY not set!")
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI")
            if "gemini" in self.model_name.lower():
                self.model_name = "gpt-4o-mini"
            logger.info(f"OpenAI provider selected with model: {self.model_name}")
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        self._initialized = True

    def get_uploaded_documents_list(self) -> list[str]:
        """Get a list of filenames currently in the data directory."""
        try:
            data_path = Path(settings.DATA_DIR)
            if not data_path.exists():
                return []
            files = []
            for item in data_path.iterdir():
                if item.is_file() and not item.name.startswith("."):
                    files.append(item.name)
            return sorted(files)
        except Exception as e:
            logger.error(f"Error listing uploaded documents: {e}")
            return []

    def get_kb_summary(self) -> str:
        """Compile a dynamic summary of topics covered by the uploaded files."""
        files = self.get_uploaded_documents_list()
        if not files:
            return "No documents are currently uploaded in the knowledge base."
            
        topics = []
        for f in files:
            name_clean = Path(f).stem.replace("_", " ").replace("-", " ").title()
            desc = "General banking information"
            lower_name = name_clean.lower()
            if "loan" in lower_name:
                desc = "Eligibility criteria, required documents, interest rates, and loan application steps."
            elif "card" in lower_name:
                desc = "Credit card types, annual fees, spend thresholds, reward structures, and interest charges."
            elif "saving" in lower_name:
                desc = "Savings account types, minimum balance requirements, and documentation."
            elif "deposit" in lower_name or "fd" in lower_name:
                desc = "Fixed deposits, interest rates, tenure options, and senior citizen benefits."
            elif "upi" in lower_name or "digital" in lower_name or "transfer" in lower_name:
                desc = "UPI transactions, NEFT/RTGS/IMPS transfer modes, timings, and daily transaction limits."
            elif "kyc" in lower_name:
                desc = "KYC documents, identity/address verification, and video KYC processes."
            elif "insurance" in lower_name:
                desc = "Insurance policies, coverage details, and claims procedures."
            
            topics.append(f"- **{name_clean}** (File: `{f}`): {desc}")
            
        return "The following documents and topics are currently available in the banking knowledge base:\n" + "\n".join(topics)

    def get_static_meta_response(self) -> str:
        """Fallback static response for metadata queries."""
        kb_summary = self.get_kb_summary()
        return (
            "Here is the current status of the banking knowledge base:\n\n"
            f"{kb_summary}\n\n"
            "Feel free to ask questions about these topics, or upload new files to expand my knowledge!"
        )

    def build_prompt(self, query: str, context: str, sources_summary: str, conversation_history: list[dict] = None) -> list[dict]:
        """
        Build the prompt with context, source citations, and conversation history.
        """
        enhanced_query = f"""## Retrieved Context (from banking knowledge base):
{context}

## Source Documents Referenced:
{sources_summary}

## Customer Question:
{query}

Provide a helpful, accurate, and well-structured response grounded ONLY in the context above. Include follow-up question suggestions at the end."""

        messages = []
        if conversation_history:
            recent_history = conversation_history[-8:]
            for msg in recent_history:
                messages.append(msg)

        messages.append({"role": "user", "content": enhanced_query})
        return messages

    async def _call_llm_with_retry(self, system_instruction: str, messages: list[dict]) -> str:
        """
        Calls the LLM provider API with retry handling and exponential backoff.
        Runs synchronous calls in a threadpool to prevent blocking the event loop.
        """
        max_retries = settings.LLM_MAX_RETRIES
        timeout = settings.LLM_TIMEOUT

        for attempt in range(max_retries):
            try:
                start_t = time.time()
                if self.provider == "gemini":
                    gemini_contents = []
                    for msg in messages:
                        role = "user" if msg["role"] == "user" else "model"
                        gemini_contents.append(
                            genai.types.Content(
                                role=role,
                                parts=[genai.types.Part(text=msg["content"])]
                            )
                        )

                    # Wrap in a threadpool to avoid blocking
                    def _call():
                        return self.client.models.generate_content(
                            model=self.model_name,
                            contents=gemini_contents,
                            config=genai.types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                temperature=0.25,
                                top_p=0.85,
                                max_output_tokens=1200,
                            )
                        )
                    response = await asyncio.to_thread(_call)
                    logger.info(f"[LLM MONITOR] Gemini call took {time.time() - start_t:.3f}s")
                    return response.text

                elif self.provider in ("groq", "openai"):
                    api_messages = [{"role": "system", "content": system_instruction}]
                    for msg in messages:
                        role = "assistant" if msg["role"] in ("assistant", "model") else "user"
                        api_messages.append({"role": role, "content": msg["content"]})

                    if self.provider == "groq":
                        url = "https://api.groq.com/openai/v1/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                            "Content-Type": "application/json"
                        }
                    else:
                        url = "https://api.openai.com/v1/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                            "Content-Type": "application/json"
                        }

                    payload = {
                        "model": self.model_name,
                        "messages": api_messages,
                        "temperature": 0.25,
                        "max_tokens": 1200
                    }

                    async with httpx.AsyncClient() as http_client:
                        response = await http_client.post(url, json=payload, headers=headers, timeout=timeout)

                    if response.status_code != 200:
                        raise Exception(f"HTTP {response.status_code}: {response.text}")

                    logger.info(f"[LLM MONITOR] {self.provider.capitalize()} call took {time.time() - start_t:.3f}s")
                    response_json = response.json()
                    return response_json["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error("LLM API calls exhausted. Bubbling exception up.")
                    raise e
                await asyncio.sleep(2 ** attempt)
        return ""

    async def _get_gemini_stream(self, system_instruction: str, contents) -> AsyncGenerator[str, None]:
        """Wrapper for Gemini stream with retry logic."""
        max_retries = settings.LLM_MAX_RETRIES
        response_stream = None
        for attempt in range(max_retries):
            try:
                response_stream = await self.client.aio.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.25,
                        top_p=0.85,
                        max_output_tokens=1200,
                    )
                )
                break
            except Exception as e:
                logger.warning(f"Gemini stream init failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)
        
        async for chunk in response_stream:
            yield chunk.text

    async def _get_openai_groq_stream(self, system_instruction: str, messages: list[dict]) -> AsyncGenerator[str, None]:
        """Wrapper for OpenAI/Groq stream with retry logic."""
        max_retries = settings.LLM_MAX_RETRIES
        timeout = settings.LLM_TIMEOUT
        
        api_messages = [{"role": "system", "content": system_instruction}]
        for msg in messages:
            role = "assistant" if msg["role"] in ("assistant", "model") else "user"
            api_messages.append({"role": role, "content": msg["content"]})

        if self.provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
        else:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

        payload = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": 0.25,
            "max_tokens": 1200,
            "stream": True
        }

        response = None
        http_client = httpx.AsyncClient()
        for attempt in range(max_retries):
            try:
                req = http_client.build_request("POST", url, json=payload, headers=headers)
                response = await http_client.send(req, stream=True, timeout=timeout)
                if response.status_code != 200:
                    body = await response.aread()
                    raise Exception(f"HTTP {response.status_code}: {body.decode()}")
                break
            except Exception as e:
                logger.warning(f"OpenAI/Groq stream init failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    await http_client.aclose()
                    raise e
                await asyncio.sleep(2 ** attempt)

        try:
            import json
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        chunk_text = data_json["choices"][0]["delta"].get("content", "")
                        if chunk_text:
                            yield chunk_text
                    except Exception as pe:
                        logger.warning(f"Error parsing SSE chunk: {pe}")
        finally:
            await response.aclose()
            await http_client.aclose()

    async def _stream_llm(self, system_instruction: str, messages: list[dict]) -> AsyncGenerator[str, None]:
        """Route to appropriate streaming generator depending on provider."""
        if self.provider == "gemini":
            gemini_contents = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                gemini_contents.append(
                    genai.types.Content(
                        role=role,
                        parts=[genai.types.Part(text=msg["content"])]
                    )
                )
            async for chunk in self._get_gemini_stream(system_instruction, gemini_contents):
                yield chunk
        else:
            async for chunk in self._get_openai_groq_stream(system_instruction, messages):
                yield chunk

    async def generate_response(
        self,
        query: str,
        conversation_history: list[dict] = None
    ) -> tuple[str, list[SourceChunk], float, list[str]]:
        """
        Generate a response using the full RAG pipeline.
        Routes queries through Greetings, Metadata, Standard RAG, and Fallback paths.
        """
        if not self._initialized:
            self.initialize()

        # Path 1: Greeting/Chat Path
        if is_greeting_or_chat(query):
            logger.info("Routing query to Greeting/Chat path.")
            messages = [{"role": "user", "content": query}]
            try:
                response_text = await self._call_llm_with_retry(CONVERSATIONAL_SYSTEM_PROMPT, messages)
                if not response_text:
                    response_text = STATIC_GREETING_RESPONSE
                clean_text, suggestions = extract_suggestions(response_text)
                if not suggestions:
                    suggestions = STATIC_GREETING_SUGGESTIONS
                return clean_text, [], 1.0, suggestions
            except Exception as e:
                logger.error(f"Error in Greeting path LLM call, falling back: {e}")
                return STATIC_GREETING_RESPONSE, [], 1.0, STATIC_GREETING_SUGREETIONS

        # Path 2: Metadata / Topics covered Query Path
        elif is_meta_query(query):
            logger.info("Routing query to Meta Query path.")
            kb_summary = self.get_kb_summary()
            messages = [{"role": "user", "content": f"{query}\n\nHere is the current KB status:\n{kb_summary}"}]
            try:
                response_text = await self._call_llm_with_retry(METADATA_SYSTEM_PROMPT, messages)
                if not response_text:
                    response_text = self.get_static_meta_response()
                clean_text, suggestions = extract_suggestions(response_text)
                if not suggestions:
                    suggestions = STATIC_META_SUGGESTIONS
                return clean_text, [], 0.9, suggestions
            except Exception as e:
                logger.error(f"Error in Meta path LLM call, falling back: {e}")
                return self.get_static_meta_response(), [], 0.9, STATIC_META_SUGGESTIONS

        # Path 3 & 4: Retrieval based RAG pipeline
        else:
            source_chunks = retriever.retrieve(query)
            context = retriever.retrieve_with_context(query)

            # Calibrate confidence score
            avg_confidence = 0.0
            if source_chunks:
                scores = [c.relevance_score for c in source_chunks]
                max_score = max(scores)
                avg_score = sum(scores) / len(scores)
                avg_confidence = round(0.7 * max_score + 0.3 * avg_score, 4)

            # Path 4: Fallback Nearest-Topics Path (Weak/No context)
            if not source_chunks:
                logger.info("Routing query to Fallback Nearest-Topics path.")
                messages = [{"role": "user", "content": query}]
                try:
                    response_text = await self._call_llm_with_retry(FALLBACK_SYSTEM_PROMPT, messages)
                    if not response_text:
                        response_text = STATIC_FALLBACK_RESPONSE
                    clean_text, suggestions = extract_suggestions(response_text)
                    if not suggestions:
                        suggestions = STATIC_FALLBACK_SUGGESTIONS
                    return clean_text, [], 0.0, suggestions
                except Exception as e:
                    logger.error(f"Error in Fallback path LLM call, falling back: {e}")
                    return STATIC_FALLBACK_RESPONSE, [], 0.0, STATIC_FALLBACK_SUGGESTIONS

            # Path 3: Standard RAG Path
            logger.info(f"Routing query to Standard RAG path with {len(source_chunks)} chunks.")
            sources_summary = "\n".join([
                f"- {c.source} (relevance: {round(c.relevance_score * 100)}%)"
                for c in source_chunks
            ])
            messages = self.build_prompt(query, context, sources_summary, conversation_history)
            try:
                response_text = await self._call_llm_with_retry(SYSTEM_PROMPT, messages)
                if not response_text:
                    response_text = "I apologize, but I wasn't able to generate a response. Please try again."
                clean_text, suggestions = extract_suggestions(response_text)
                return clean_text, source_chunks, avg_confidence, suggestions
            except Exception as e:
                logger.error(f"Error in Standard RAG path LLM call: {e}")
                err_response = (
                    f"I apologize, but I encountered an error while communicating with the AI model. "
                    f"Here is some information retrieved from our files:\n\n{context[:300]}..."
                )
                return err_response, source_chunks, avg_confidence, []

    async def generate_response_stream(
        self,
        query: str,
        conversation_history: list[dict] = None
    ) -> AsyncGenerator[dict, None]:
        """
        Generate a response stream using the full RAG pipeline.
        Yields dictionaries representing SSE events.
        """
        if not self._initialized:
            self.initialize()

        # Path 1: Greeting/Chat Path
        if is_greeting_or_chat(query):
            logger.info("Routing stream query to Greeting/Chat path.")
            yield {
                "event": "metadata",
                "sources": [],
                "confidence": 1.0
            }
            messages = [{"role": "user", "content": query}]
            
            accumulated_text = ""
            yielded_len = 0
            try:
                async for chunk_text in self._stream_llm(CONVERSATIONAL_SYSTEM_PROMPT, messages):
                    if not chunk_text:
                        continue
                    accumulated_text += chunk_text
                    
                    # Buffer check for suggestions block
                    lower_accum = accumulated_text.lower()
                    indicator_index = -1
                    for term in ["💡", "you might also", "suggested questions"]:
                        idx = lower_accum.find(term)
                        if idx != -1:
                            if indicator_index == -1 or idx < indicator_index:
                                indicator_index = idx

                    if indicator_index == -1:
                        ends_with_partial = False
                        for term in ["💡", "you might also", "suggested questions"]:
                            for i in range(1, len(term)):
                                if term.startswith(lower_accum[-i:]):
                                    ends_with_partial = True
                                    break
                            if ends_with_partial:
                                break
                        
                        if ends_with_partial:
                            safe_len = max(0, len(accumulated_text) - 25)
                            if safe_len > yielded_len:
                                to_yield = accumulated_text[yielded_len:safe_len]
                                yielded_len = safe_len
                                if to_yield:
                                    yield {"event": "token", "text": to_yield}
                        else:
                            to_yield = accumulated_text[yielded_len:]
                            yielded_len = len(accumulated_text)
                            if to_yield:
                                yield {"event": "token", "text": to_yield}
                    else:
                        if indicator_index > yielded_len:
                            to_yield = accumulated_text[yielded_len:indicator_index]
                            yielded_len = indicator_index
                            if to_yield:
                                yield {"event": "token", "text": to_yield}

                clean_response, suggestions = extract_suggestions(accumulated_text)
                if not suggestions:
                    suggestions = STATIC_GREETING_SUGGESTIONS
                
                if len(clean_response) > yielded_len:
                    to_yield = clean_response[yielded_len:]
                    if to_yield:
                        yield {"event": "token", "text": to_yield}
                        
                yield {
                    "event": "done",
                    "text": clean_response,
                    "suggested_questions": suggestions
                }
            except Exception as e:
                logger.error(f"Error streaming greeting: {e}")
                words = STATIC_GREETING_RESPONSE.split(" ")
                for i, word in enumerate(words):
                    token = word if i == 0 else " " + word
                    yield {"event": "token", "text": token}
                    await asyncio.sleep(0.02)
                yield {
                    "event": "done",
                    "text": STATIC_GREETING_RESPONSE,
                    "suggested_questions": STATIC_GREETING_SUGGESTIONS
                }

        # Path 2: Metadata / Topics covered Query Path
        elif is_meta_query(query):
            logger.info("Routing stream query to Meta Query path.")
            yield {
                "event": "metadata",
                "sources": [],
                "confidence": 0.9
            }
            kb_summary = self.get_kb_summary()
            messages = [{"role": "user", "content": f"{query}\n\nHere is the current KB status:\n{kb_summary}"}]
            
            accumulated_text = ""
            yielded_len = 0
            try:
                async for chunk_text in self._stream_llm(METADATA_SYSTEM_PROMPT, messages):
                    if not chunk_text:
                        continue
                    accumulated_text += chunk_text
                    
                    # Buffer check
                    lower_accum = accumulated_text.lower()
                    indicator_index = -1
                    for term in ["💡", "you might also", "suggested questions"]:
                        idx = lower_accum.find(term)
                        if idx != -1:
                            if indicator_index == -1 or idx < indicator_index:
                                indicator_index = idx

                    if indicator_index == -1:
                        ends_with_partial = False
                        for term in ["💡", "you might also", "suggested questions"]:
                            for i in range(1, len(term)):
                                if term.startswith(lower_accum[-i:]):
                                    ends_with_partial = True
                                    break
                            if ends_with_partial:
                                break
                        
                        if ends_with_partial:
                            safe_len = max(0, len(accumulated_text) - 25)
                            if safe_len > yielded_len:
                                to_yield = accumulated_text[yielded_len:safe_len]
                                yielded_len = safe_len
                                if to_yield:
                                    yield {"event": "token", "text": to_yield}
                        else:
                            to_yield = accumulated_text[yielded_len:]
                            yielded_len = len(accumulated_text)
                            if to_yield:
                                yield {"event": "token", "text": to_yield}
                    else:
                        if indicator_index > yielded_len:
                            to_yield = accumulated_text[yielded_len:indicator_index]
                            yielded_len = indicator_index
                            if to_yield:
                                yield {"event": "token", "text": to_yield}

                clean_response, suggestions = extract_suggestions(accumulated_text)
                if not suggestions:
                    suggestions = STATIC_META_SUGGESTIONS
                
                if len(clean_response) > yielded_len:
                    to_yield = clean_response[yielded_len:]
                    if to_yield:
                        yield {"event": "token", "text": to_yield}

                yield {
                    "event": "done",
                    "text": clean_response,
                    "suggested_questions": suggestions
                }
            except Exception as e:
                logger.error(f"Error streaming meta response: {e}")
                static_meta = self.get_static_meta_response()
                words = static_meta.split(" ")
                for i, word in enumerate(words):
                    token = word if i == 0 else " " + word
                    yield {"event": "token", "text": token}
                    await asyncio.sleep(0.02)
                yield {
                    "event": "done",
                    "text": static_meta,
                    "suggested_questions": STATIC_META_SUGGESTIONS
                }

        # Path 3 & 4: Retrieval based RAG pipeline
        else:
            source_chunks = retriever.retrieve(query)
            context = retriever.retrieve_with_context(query)

            # Calibrate confidence score
            avg_confidence = 0.0
            if source_chunks:
                scores = [c.relevance_score for c in source_chunks]
                max_score = max(scores)
                avg_score = sum(scores) / len(scores)
                avg_confidence = round(0.7 * max_score + 0.3 * avg_score, 4)

            yield {
                "event": "metadata",
                "sources": [c.dict() for c in source_chunks],
                "confidence": avg_confidence
            }

            # Path 4: Fallback Nearest-Topics Path (Weak/No context)
            if not source_chunks:
                logger.info("Routing stream query to Fallback Nearest-Topics path.")
                messages = [{"role": "user", "content": query}]
                
                accumulated_text = ""
                yielded_len = 0
                try:
                    async for chunk_text in self._stream_llm(FALLBACK_SYSTEM_PROMPT, messages):
                        if not chunk_text:
                            continue
                        accumulated_text += chunk_text
                        
                        # Buffer check
                        lower_accum = accumulated_text.lower()
                        indicator_index = -1
                        for term in ["💡", "you might also", "suggested questions"]:
                            idx = lower_accum.find(term)
                            if idx != -1:
                                if indicator_index == -1 or idx < indicator_index:
                                    indicator_index = idx

                        if indicator_index == -1:
                            ends_with_partial = False
                            for term in ["💡", "you might also", "suggested questions"]:
                                for i in range(1, len(term)):
                                    if term.startswith(lower_accum[-i:]):
                                        ends_with_partial = True
                                        break
                                if ends_with_partial:
                                    break
                            
                            if ends_with_partial:
                                safe_len = max(0, len(accumulated_text) - 25)
                                if safe_len > yielded_len:
                                    to_yield = accumulated_text[yielded_len:safe_len]
                                    yielded_len = safe_len
                                    if to_yield:
                                        yield {"event": "token", "text": to_yield}
                            else:
                                to_yield = accumulated_text[yielded_len:]
                                yielded_len = len(accumulated_text)
                                if to_yield:
                                    yield {"event": "token", "text": to_yield}
                        else:
                            if indicator_index > yielded_len:
                                to_yield = accumulated_text[yielded_len:indicator_index]
                                yielded_len = indicator_index
                                if to_yield:
                                    yield {"event": "token", "text": to_yield}

                    clean_response, suggestions = extract_suggestions(accumulated_text)
                    if not suggestions:
                        suggestions = STATIC_FALLBACK_SUGGESTIONS
                    
                    if len(clean_response) > yielded_len:
                        to_yield = clean_response[yielded_len:]
                        if to_yield:
                            yield {"event": "token", "text": to_yield}

                    yield {
                        "event": "done",
                        "text": clean_response,
                        "suggested_questions": suggestions
                    }
                except Exception as e:
                    logger.error(f"Error streaming fallback response: {e}")
                    words = STATIC_FALLBACK_RESPONSE.split(" ")
                    for i, word in enumerate(words):
                        token = word if i == 0 else " " + word
                        yield {"event": "token", "text": token}
                        await asyncio.sleep(0.02)
                    yield {
                        "event": "done",
                        "text": STATIC_FALLBACK_RESPONSE,
                        "suggested_questions": STATIC_FALLBACK_SUGGESTIONS
                    }
                return

            # Path 3: Standard RAG Path
            logger.info(f"Routing stream query to Standard RAG path with {len(source_chunks)} chunks.")
            sources_summary = "\n".join([
                f"- {c.source} (relevance: {round(c.relevance_score * 100)}%)"
                for c in source_chunks
            ])
            messages = self.build_prompt(query, context, sources_summary, conversation_history)
            
            accumulated_text = ""
            yielded_len = 0
            try:
                async for chunk_text in self._stream_llm(SYSTEM_PROMPT, messages):
                    if not chunk_text:
                        continue
                    accumulated_text += chunk_text
                    
                    # Buffer check
                    lower_accum = accumulated_text.lower()
                    indicator_index = -1
                    for term in ["💡", "you might also", "suggested questions"]:
                        idx = lower_accum.find(term)
                        if idx != -1:
                            if indicator_index == -1 or idx < indicator_index:
                                indicator_index = idx

                    if indicator_index == -1:
                        ends_with_partial = False
                        for term in ["💡", "you might also", "suggested questions"]:
                            for i in range(1, len(term)):
                                if term.startswith(lower_accum[-i:]):
                                    ends_with_partial = True
                                    break
                            if ends_with_partial:
                                break
                        
                        if ends_with_partial:
                            safe_len = max(0, len(accumulated_text) - 25)
                            if safe_len > yielded_len:
                                to_yield = accumulated_text[yielded_len:safe_len]
                                yielded_len = safe_len
                                if to_yield:
                                    yield {"event": "token", "text": to_yield}
                        else:
                            to_yield = accumulated_text[yielded_len:]
                            yielded_len = len(accumulated_text)
                            if to_yield:
                                yield {"event": "token", "text": to_yield}
                    else:
                        if indicator_index > yielded_len:
                            to_yield = accumulated_text[yielded_len:indicator_index]
                            yielded_len = indicator_index
                            if to_yield:
                                yield {"event": "token", "text": to_yield}

                clean_response, suggestions = extract_suggestions(accumulated_text)
                
                if len(clean_response) > yielded_len:
                    to_yield = clean_response[yielded_len:]
                    if to_yield:
                        yield {"event": "token", "text": to_yield}

                yield {
                    "event": "done",
                    "text": clean_response,
                    "suggested_questions": suggestions
                }
            except Exception as e:
                logger.error(f"Error streaming standard RAG response: {e}")
                err_response = (
                    f"I apologize, but I encountered an error while communicating with the AI model. "
                    f"Here is some information retrieved from our files:\n\n{context[:300]}..."
                )
                yield {"event": "token", "text": err_response}
                yield {
                    "event": "done",
                    "text": err_response,
                    "suggested_questions": []
                }


# Singleton instance
generator = ResponseGenerator()
