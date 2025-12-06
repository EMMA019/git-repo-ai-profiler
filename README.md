🤖 git-repo-ai-profiler

Analyze your Git history with Google Gemini AI.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://git-repo-ai-profiler-gjtbz72whwbjretxnkv25s.streamlit.app/)


🧐 What is this?

Git Repository AI Profiler is a Streamlit application that mines your Git repository meta-data and uses Google Gemini 2.5 Flash to profile your development habits.

Instead of boring statistics, it gives you a "Developer Persona" (e.g., "The Midnight Sprinter", "The Weekend Warrior") and provides concrete, sometimes harsh, advice to improve your code quality and work-life balance.

✨ Key Features

📊 Interactive Visualizations:

Monthly Commits: Track your productivity trends.

Activity Heatmap: Visualize your peak coding hours (Day vs Hour).

Churn Ranking: Identify "High-Risk Files" that are modified too frequently.

🧠 AI-Powered Profiling (in Japanese):

Generates a unique "Dev Persona" based on your commit patterns.

Provides CTO-level advice on refactoring, burnout prevention, and architectural improvements.

⚡ High Performance:

Real-time progress tracking for long mining tasks.

Robust API handling with automatic retries.

🔒 Secure:

Users can input their own API Key securely via the sidebar.

🚀 Quick Start

1. Clone the repository

git clone [https://github.com/EMMA019/git-repo-ai-profiler.git](https://github.com/EMMA019/git-repo-ai-profiler.git)
cd git-repo-ai-profiler



2. Set up the environment

It is recommended to use a virtual environment.

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate



3. Install dependencies

pip install -r requirements.txt



4. Get your API Key

You need a Google Gemini API Key. Get it from Google AI Studio.

5. Run the App!

streamlit run app.py



🛠️ Tech Stack

Frontend: Streamlit

Data Processing: Pandas, PyDriller

Visualization: Plotly

AI Engine: Google Gemini API (via google-generativeai)

Resilience: Tenacity

🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Fork the project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License

Distributed under the MIT License. See LICENSE for more information.

Created with ❤️ and AI
