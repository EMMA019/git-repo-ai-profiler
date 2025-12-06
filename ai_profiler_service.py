import os
import streamlit as st
import google.generativeai as genai

class AIProfilerService:
    """
    A service class to interact with the Google Gemini AI model for
    generating analysis of development styles based on repository data.
    """
    def __init__(self):
        """
        Initializes the Gemini API configuration and model.
        Requires the 'GOOGLE_API_KEY' environment variable to be set.
        If the API key is not found, the Streamlit application will stop.
        """
        # Retrieve the Google API key from environment variables
        api_key = os.getenv("GOOGLE_API_KEY")

        # Check if the API key is provided
        if not api_key:
            st.error("GOOGLE_API_KEY environment variable is not set. "
                     "Please configure your Google Gemini API key to use AI analysis features.")
            st.stop()  # Halt the Streamlit application if the key is missing
        
        # Configure the generative AI library with the retrieved API key
        genai.configure(api_key=api_key)
        
        # Initialize the Gemini model.
        # 'gemini-1.5-flash' is chosen for its efficiency, speed, and capability
        # in summarization and analytical tasks, making it suitable for profiling
        # development styles without excessive latency.
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # @st.cache_data(ttl=3600, show_spinner="Getting AI analysis from Gemini...")
    def get_ai_analysis(self, prompt_text: str) -> str:
        """
        Calls the configured Google Gemini model to generate a comprehensive analysis
        of the development style based on the provided detailed prompt text.
        This function leverages Streamlit's caching mechanism (`st.cache_data`)
        to store results for 1 hour (3600 seconds), significantly improving
        performance and reducing redundant API calls for identical inputs.

        Args:
            prompt_text (str): The meticulously prepared text prompt containing
                                a summary of the repository's commit patterns,
                                activity heatmap insights, file extension changes,
                                and other relevant data points. This text serves
                                as the input for Gemini to generate its analysis.

        Returns:
            str: A string containing the AI-generated analysis of the development style.
                 If the API call fails or Gemini returns an unexpected response,
                 a user-friendly error message will be returned instead.
        """
        try:
            # Generate content using the initialized Gemini model
            response = self.model.generate_content(prompt_text)
            
            # Validate the response from Gemini to ensure it contains content
            if not response.candidates:
                return "Gemini returned no valid candidates for analysis. " \
                       "This might indicate an issue with the prompt or the model's response."
            
            # Extract and return the generated text from the first candidate
            return response.text
        except Exception as e:
            # Catch any exceptions that may occur during the API call (e.g., network errors,
            # authentication failures, rate limits, or other API-specific errors).
            st.error(f"An error occurred while calling the Gemini API: {e}")
            return "Failed to retrieve AI analysis due to an internal error or API issue. " \
                   "Please verify your API key, network connection, and try again."