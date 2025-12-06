import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import logging
import re
from collections import defaultdict
import shutil
import tempfile
import subprocess

# Import services
from git_miner_service import mine_git_repository
from data_analyzer_service import DataAnalyzerService
from ai_profiler_service import AIProfilerService

# For tenacity (retry logic)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Plotting Functions ---

def plot_monthly_commits(monthly_commit_data):
    """Plots the number of commits per month."""
    if monthly_commit_data.empty:
        st.write("No commit data available to plot monthly commits.")
        return None

    fig = px.bar(
        monthly_commit_data,
        x='Month',
        y='Commits',
        title='Monthly Commit Activity',
        labels={'Commits': 'Number of Commits', 'Month': 'Month'},
        hover_data={'Month': '|%Y-%m'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_xaxes(tickformat='%Y-%m')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_activity_heatmap(activity_heatmap_data):
    """Plots a heatmap of commit activity by hour and weekday."""
    if activity_heatmap_data.empty:
        st.write("No activity heatmap data available.")
        return None

    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    all_hours = range(24)
    df_full = pd.DataFrame([(d, h) for d in weekdays for h in all_hours], columns=['Day of Week', 'Hour of Day'])
    activity_heatmap_data_merged = pd.merge(df_full, activity_heatmap_data, on=['Day of Week', 'Hour of Day'], how='left').fillna(0)
    activity_heatmap_data_merged['Commits'] = activity_heatmap_data_merged['Commits'].astype(int)

    activity_heatmap_data_merged['Day of Week'] = pd.Categorical(
        activity_heatmap_data_merged['Day of Week'],
        categories=weekdays,
        ordered=True
    )
    activity_heatmap_data_merged = activity_heatmap_data_merged.sort_values(['Day of Week', 'Hour of Day'])

    fig = px.density_heatmap(
        activity_heatmap_data_merged,
        x='Hour of Day',
        y='Day of Week',
        z='Commits',
        title='Commit Activity Heatmap (Hour of Day vs. Day of Week)',
        labels={'Hour of Day': 'Hour of Day', 'Day of Week': 'Day of Week', 'Commits': 'Number of Commits'},
        color_continuous_scale="Viridis",
        category_orders={"Day of Week": weekdays}
    )
    fig.update_xaxes(side="top", tickvals=list(range(24)))
    fig.update_layout(yaxis_autorange="reversed")
    return fig

def plot_file_extension_changes(file_extension_data):
    """Plots the distribution of file extension changes."""
    if file_extension_data.empty:
        st.write("No file extension data available.")
        return None

    fig = px.bar(
        file_extension_data.head(10),
        x='Extension',
        y='Changes',
        title='File Extension Total Changes (Top 10)',
        labels={'Extension': 'File Extension', 'Changes': 'Total Lines Changed'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_file_churn_ranking(file_churn_data):
    """Plots a bar chart of the top N files by churn."""
    if file_churn_data.empty:
        st.write("No file churn data available.")
        return None

    max_files = min(20, len(file_churn_data))
    if max_files == 0:
        st.write("No file churn data available to display.")
        return None
    
    top_n = st.slider(
        "Number of files to show in Churn Ranking:",
        min_value=5,
        max_value=max_files,
        value=min(10, max_files)
    )
    display_data = file_churn_data.head(top_n)

    fig = px.bar(
        display_data,
        x='churn_count',
        y='file_path',
        orientation='h',
        title=f'Top {top_n} Files by Churn (Most Frequent Changes)',
        labels={'churn_count': 'Number of Commits Affecting File', 'file_path': 'File Path'},
        color_discrete_sequence=px.colors.qualitative.Vivid,
        height=min(600, 50 * top_n + 150)
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

# --- Analysis Orchestration Function ---

# @st.cache_data(show_spinner=False)
def run_analysis(repo_url, repo_path, progress_callback, api_key):
    """
    Orchestrates the git mining, data analysis, and AI profiling.
    Uses a progress callback for Streamlit.
    """
    # Pass the user's API key to the service
    ai_profiler = AIProfilerService(user_key=api_key)
    
    results = {}
    temp_repo_dir = None

    try:
        current_repo_path = repo_path
        
        if repo_url and not repo_path:
            progress_callback(5, f"Cloning repository from {repo_url}...")
            logger.info(f"Cloning GitHub repository: {repo_url}")
            temp_repo_dir = tempfile.mkdtemp()
            
            try:
                subprocess.run(['git', 'clone', '--depth', '100', repo_url, temp_repo_dir], check=True)
                current_repo_path = temp_repo_dir
                logger.info(f"Repository cloned to temporary directory: {current_repo_path}")
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to clone repository from URL: {repo_url}. Please ensure the URL is correct and the repository is public. Error: {e}")
                logger.error(f"Failed to clone repository: {e}", exc_info=True)
                return None
            except Exception as e:
                st.error(f"An unexpected error occurred during cloning: {e}")
                logger.error(f"Cloning failed: {e}", exc_info=True)
                return None
        
        if not current_repo_path:
            st.error("No valid repository path determined for analysis.")
            return None

        # Step 1: Mine Git Repository
        progress_callback(10, "Mining Git Repository...")
        logger.info(f"Mining repository: Path={current_repo_path}")
        unique_commits_df, file_modifications_df = mine_git_repository(
            repo_path=current_repo_path,
            progress_callback=lambda current, total, msg: progress_callback(10 + int(current/total*20), msg)
        )
        
        if unique_commits_df.empty:
            st.error("No commit data found for the provided repository.")
            return None

        analyzer = DataAnalyzerService(unique_commits_df, file_modifications_df)

        # Step 2-6: Prepare Data
        progress_callback(35, "Analyzing Monthly Commits...")
        monthly_commit_data = analyzer.prepare_monthly_commit_data()
        results['monthly_commit_data'] = monthly_commit_data

        progress_callback(45, "Analyzing Commit Activity Heatmap...")
        activity_heatmap_data = analyzer.prepare_activity_heatmap_data()
        results['activity_heatmap_data'] = activity_heatmap_data

        progress_callback(60, "Analyzing File Extension Changes...")
        file_extension_data = analyzer.prepare_file_extension_data()
        results['file_extension_data'] = file_extension_data

        progress_callback(75, "Analyzing File Churn Ranking...")
        file_churn_data = analyzer.prepare_file_churn_ranking_data()
        results['file_churn_data'] = file_churn_data

        progress_callback(85, "Generating Analysis Summary for AI...")
        analysis_summary_dict = analyzer.generate_analysis_summary()
        analysis_summary = analysis_summary_dict['summary_text']
        results['analysis_summary'] = analysis_summary
        
        # Step 7: Get AI Analysis with Retry Logic
        progress_callback(90, "Calling AI for deep analysis (this may take a minute)...")
        logger.info("Calling AI profiler service...")

        # Update prompt for better actionable advice
        enhanced_prompt = f"""
        Based on the following Git repository analysis summary, please provide a "Development Style Diagnosis" and "Concrete Advice".
        
        Analysis Summary:
        {analysis_summary}

        Please output in the following format (Japanese):
        
        ### 開発スタイルの総評：【(Catchy Title)】
        (Description of the style based on data)

        ### 改善のための実用的なアドバイス
        1. **[Point 1]**: [Actionable advice]
        2. **[Point 2]**: [Actionable advice]
        3. **[Point 3]**: [Actionable advice]
        """

        ai_analysis_with_retry = retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying AI analysis (attempt {retry_state.attempt_number}/{retry_state.max_attempts_reached + 1})..."
            )
        )(ai_profiler.get_ai_analysis)

        try:
            ai_summary_results = ai_analysis_with_retry(enhanced_prompt)
            results['ai_summary_results'] = ai_summary_results
        except Exception as e:
            logger.error(f"Failed to get AI analysis after multiple retries: {e}", exc_info=True)
            st.error(f"Failed to get AI analysis. Please check your API key. Error: {e}")
            results['ai_summary_results'] = "AI analysis failed."

        progress_callback(100, "Analysis complete!")
        return results

    except Exception as e:
        logger.error(f"An unexpected error occurred during analysis: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during analysis: {e}")
        return None
    finally:
        if temp_repo_dir and os.path.exists(temp_repo_dir):
            try:
                shutil.rmtree(temp_repo_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary cloned repository at: {temp_repo_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary repository {temp_repo_dir}: {e}")

# --- Main Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="Git Repository AI Profiler")

    st.title("🤖 Git Repository AI Profiler")

    with st.sidebar:
        st.header("Settings")
        
        # --- API Key Input ---
        user_api_key = st.text_input("Enter your Gemini API Key", type="password", help="Get your key from https://aistudio.google.com/app/apikey")
        
        st.markdown("---")
        
        st.header("Repository Input")
        repo_option = st.radio(
            "Select repository source:",
            ("Enter GitHub URL", "Enter Local Path")
        )

        repo_path_for_analysis = None
        repo_url_for_analysis = None

        if repo_option == "Enter Local Path":
            local_path_input = st.text_input("Enter path to local Git repository", value=os.getcwd())
            if os.path.isdir(local_path_input) and os.path.exists(os.path.join(local_path_input, '.git')):
                repo_path_for_analysis = local_path_input
                st.success("Valid local Git repository found.")
            elif local_path_input:
                st.warning("Invalid Git repository path.")

        elif repo_option == "Enter GitHub URL":
            github_url = st.text_input("Enter GitHub repository URL", value="https://github.com/streamlit/streamlit")
            if github_url:
                repo_url_for_analysis = github_url

        st.subheader("Analysis Controls")
        
        # Check if API Key is present
        # Strictly require user input, ignoring environment variables to prevent usage of the developer's key
        has_api_key = bool(user_api_key)
        
        if st.button("Analyze Repository", type="primary", disabled=not (bool(repo_path_for_analysis) or bool(repo_url_for_analysis))):
            if not has_api_key:
                st.error("🔒 Please enter YOUR Gemini API Key in the sidebar. This app requires your own key to function.")
            else:
                st.session_state['run_analysis'] = True
                st.session_state['repo_path_param'] = repo_path_for_analysis
                st.session_state['repo_url_param'] = repo_url_for_analysis
                st.session_state['user_api_key'] = user_api_key # Store for this session
                st.session_state['analysis_results'] = None
        
        if st.button("Clear Cache & Reset"):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()

    # --- Main Content Area ---
    if 'run_analysis' in st.session_state and st.session_state['run_analysis']:
        # Double check API key before running
        current_api_key = st.session_state.get('user_api_key')
        
        display_repo_info = st.session_state.get('repo_url_param') or st.session_state.get('repo_path_param')
        st.info(f"Starting analysis for repository: {display_repo_info}")
        
        progress_text_placeholder = st.empty()
        progress_bar_placeholder = st.progress(0)

        def update_progress_ui(percent_complete, message):
            progress_bar_placeholder.progress(int(percent_complete) / 100)
            progress_text_placeholder.text(f"Progress: {message} ({int(percent_complete)}%)")

        results = run_analysis(
            repo_url=st.session_state.get('repo_url_param'),
            repo_path=st.session_state.get('repo_path_param'),
            progress_callback=update_progress_ui,
            api_key=current_api_key # Pass the key
        )
        st.session_state['analysis_results'] = results
        st.session_state['run_analysis'] = False
        st.rerun()

    if 'analysis_results' in st.session_state and st.session_state['analysis_results'] is not None:
        results = st.session_state['analysis_results']

        st.success("Analysis Complete!")

        st.subheader("AI-Powered Repository Insights")
        if 'ai_summary_results' in results and results['ai_summary_results'] and results['ai_summary_results'] != "AI analysis failed.":
            st.markdown(results['ai_summary_results'])
        else:
            st.warning("AI analysis results are not available.")

        st.subheader("Detailed Repository Metrics")
        tab_titles = ["Monthly Commits", "Activity Heatmap", "File Extension Changes", "File Churn Ranking"]
        tabs = st.tabs(tab_titles)

        with tabs[0]:
            if 'monthly_commit_data' in results:
                st.plotly_chart(plot_monthly_commits(results['monthly_commit_data']), use_container_width=True)
        with tabs[1]:
            if 'activity_heatmap_data' in results:
                st.plotly_chart(plot_activity_heatmap(results['activity_heatmap_data']), use_container_width=True)
        with tabs[2]:
            if 'file_extension_data' in results:
                st.plotly_chart(plot_file_extension_changes(results['file_extension_data']), use_container_width=True)
        with tabs[3]:
            if 'file_churn_data' in results:
                st.plotly_chart(plot_file_churn_ranking(results['file_churn_data']), use_container_width=True)

    else:
        st.info("👈 Enter your Gemini API Key and Repository details in the sidebar to start.")

if __name__ == "__main__":
    main()