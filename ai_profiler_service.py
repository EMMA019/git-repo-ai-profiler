import os
import streamlit as st
import google.generativeai as genai

class AIProfilerService:
    """
    A service class to interact with the Google Gemini AI model for
    generating analysis of development styles based on repository data.
    """
    def __init__(self, user_key=None):
        """
        Initializes the Gemini API configuration and model.
        Strictly requires a user-provided key. Does NOT fallback to environment variables
        to prevent accidental usage of the developer's quota.
        """
        # Strictly use the key provided by the user via the UI
        api_key = user_key

        # Check if the API key is provided
        if not api_key:
            st.error("Gemini API Key is missing. Please enter it in the sidebar settings.")
            st.stop()  # Halt the Streamlit application if the key is missing
        
        # Configure the generative AI library with the retrieved API key
        genai.configure(api_key=api_key)
        
        # Initialize the Gemini model.
        # 'gemini-1.5-flash' is chosen for its efficiency, speed, and capability
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # @st.cache_data(ttl=3600, show_spinner="Getting AI analysis from Gemini...")
    def get_ai_analysis(self, prompt_text: str) -> str:
        """
        Calls the configured Google Gemini model to generate a comprehensive analysis
        of the development style based on the provided detailed prompt text.
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
            st.error(f"An error occurred while calling the Gemini API: {e}")
            return "Failed to retrieve AI analysis due to an internal error or API issue. " \
                   "Please verify your API key, network connection, and try again."