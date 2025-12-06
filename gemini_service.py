import os
import google.generativeai as genai
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import logging

# Configure logging for tenacity and Gemini service
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiService:
    """
    A service class to interact with the Google Gemini API,
    incorporating retry logic for robust API calls.
    """

    def __init__(self):
        """
        Initializes the GeminiService by configuring the API key
        and loading the Gemini Pro model.
        Raises a ValueError if the GEMINI_API_KEY environment variable is not set.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set. Please set it to use Gemini.")
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        logger.info("GeminiService initialized and model 'gemini-pro' loaded.")

    @retry(
        wait=wait_random_exponential(multiplier=1, min=4, max=10), # Exponential backoff with random jitter
        stop=stop_after_attempt(3),                              # Stop after 3 attempts
        retry=retry_if_exception_type(Exception),                 # Retry on any exception
        reraise=True                                             # Re-raise the last exception if all retries fail
    )
    def generate_content_with_retry(self, prompt: str) -> str:
        """
        Sends a prompt to the Gemini API and retrieves the generated content,
        with built-in retry logic using tenacity.

        Args:
            prompt: The text prompt to send to the Gemini model.

        Returns:
            The generated text content from the Gemini model.

        Raises:
            Exception: If the Gemini API call fails after all retries,
                       or if the response content is empty/invalid.
        """
        logger.info(f"Attempting Gemini content generation (attempt {self.generate_content_with_retry.retry.statistics['attempts'] + 1} of 3). Prompt snippet: {prompt[:100]}...")
        try:
            response = self.model.generate_content(prompt)

            # Check if the response contains valid parts and text
            if not response.parts or not response.text:
                logger.warning(f"Gemini API returned an empty or invalid response. Response: {response}")
                # Raise an error to trigger a retry if applicable
                raise ValueError("Gemini API returned no content or invalid response.")

            logger.info("Gemini content generation successful.")
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed with error: {e}")
            # The 'reraise=True' in @retry decorator will handle re-raising after all attempts.
            # We just need to ensure an exception is raised here to signal failure to tenacity.
            raise