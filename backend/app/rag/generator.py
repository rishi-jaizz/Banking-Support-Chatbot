"""
LLM response generation module.
Uses Google Gemini to generate context-aware responses based on retrieved documents.
"""

import logging
from google import genai
from app.config import settings
from app.rag.retriever import retriever
from app.models.schemas import SourceChunk

logger = logging.getLogger(__name__)

# System prompt that guides the LLM's behavior
SYSTEM_PROMPT = """You are an expert Banking Support Assistant for an Indian bank. Your role is to help customers with banking-related queries by providing accurate, helpful, and professional responses.

## Guidelines:
1. **Use ONLY the provided context** to answer questions. Do not make up information.
2. If the context doesn't contain enough information, say so honestly and suggest the customer contact their bank directly.
3. Be professional, friendly, and empathetic in your tone.
4. Provide specific details like interest rates, fees, limits, and eligibility criteria when available in the context.
5. Use bullet points and structured formatting for clarity.
6. For complex queries, break down the answer into clear steps.
7. Always mention important disclaimers (e.g., "rates may vary by bank", "subject to eligibility").
8. If asked about something completely unrelated to banking, politely redirect the conversation.
9. Keep responses concise but comprehensive — aim for helpful, not lengthy.
10. When citing specific numbers or policies, note that these are general guidelines and actual values may vary.

## Important:
- You are NOT authorized to perform any banking transactions.
- Do not ask for or store any personal/financial information.
- Always recommend customers verify details with their specific bank."""


class ResponseGenerator:
    """
    Generates context-aware responses using Google Gemini LLM.
    Combines retrieved context with conversation history for coherent responses.
    """

    def __init__(self):
        self.client = None
        self.model_name = settings.LLM_MODEL

    def initialize(self):
        """Initialize the Gemini client."""
        if not settings.GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY not set!")
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        logger.info(f"Gemini client initialized with model: {self.model_name}")

    def build_prompt(self, query: str, context: str, conversation_history: list[dict] = None) -> list[dict]:
        """
        Build the prompt with context and conversation history.

        Args:
            query: Current user question.
            context: Retrieved document context.
            conversation_history: Previous messages in the conversation.

        Returns:
            List of message dicts for the Gemini API.
        """
        # Build the context-enhanced user message
        enhanced_query = f"""## Retrieved Banking Knowledge (Context):
{context}

## Customer Question:
{query}

Please provide a helpful, accurate response based on the context above. If the context doesn't fully address the question, acknowledge the limitation."""

        messages = []

        # Add conversation history (last 6 messages for context window management)
        if conversation_history:
            recent_history = conversation_history[-6:]
            for msg in recent_history:
                messages.append(msg)

        # Add the current query with context
        messages.append({"role": "user", "content": enhanced_query})

        return messages

    async def generate_response(
        self,
        query: str,
        conversation_history: list[dict] = None
    ) -> tuple[str, list[SourceChunk]]:
        """
        Generate a response using RAG pipeline:
        1. Retrieve relevant context
        2. Build prompt with context + history
        3. Generate response with Gemini

        Args:
            query: User's question.
            conversation_history: Previous messages for context.

        Returns:
            Tuple of (response_text, source_chunks).
        """
        if self.client is None:
            self.initialize()

        # Step 1: Retrieve relevant context
        source_chunks = retriever.retrieve(query)
        context = retriever.retrieve_with_context(query)

        # Step 2: Build prompt
        messages = self.build_prompt(query, context, conversation_history)

        # Step 3: Generate response with Gemini
        try:
            # Convert messages to Gemini format
            gemini_contents = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                gemini_contents.append(
                    genai.types.Content(
                        role=role,
                        parts=[genai.types.Part(text=msg["content"])]
                    )
                )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=gemini_contents,
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    top_p=0.9,
                    max_output_tokens=1024,
                )
            )

            response_text = response.text

            if not response_text:
                response_text = "I apologize, but I wasn't able to generate a response. Please try rephrasing your question or contact customer support for assistance."

            logger.info(f"Generated response for query: '{query[:50]}...'")
            return response_text, source_chunks

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_msg = (
                "I'm sorry, I encountered an error while processing your request. "
                "Please try again in a moment or contact customer support for immediate assistance."
            )
            return error_msg, source_chunks


# Singleton instance
generator = ResponseGenerator()
