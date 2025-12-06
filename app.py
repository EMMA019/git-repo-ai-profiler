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
import shutil # Added for temporary directory cleanup
import tempfile # Added for temporary directory creation for cloning
import subprocess # Added for git clone command

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
    """
    Plots the number of commits per month.
    """
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
    """
    Plots a heatmap of commit activity by hour and weekday.
    """
    if activity_heatmap_data.empty:
        st.write("No activity heatmap data available.")
        return None

    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Ensure all weekdays and hours are present to avoid gaps in heatmap
    all_hours = range(24)
    # Corrected column names to match output from DataAnalyzerService.prepare_activity_heatmap_data
    df_full = pd.DataFrame([(d, h) for d in weekdays for h in all_hours], columns=['Day of Week', 'Hour of Day'])
    activity_heatmap_data_merged = pd.merge(df_full, activity_heatmap_data, on=['Day of Week', 'Hour of Day'], how='left').fillna(0)
    activity_heatmap_data_merged['Commits'] = activity_heatmap_data_merged['Commits'].astype(int)

    # Sort weekdays for plotting
    activity_heatmap_data_merged['Day of Week'] = pd.Categorical(
        activity_heatmap_data_merged['Day of Week'],
        categories=weekdays,
        ordered=True
    )
    activity_heatmap_data_merged = activity_heatmap_data_merged.sort_values(['Day of Week', 'Hour of Day'])

    fig = px.density_heatmap(
        activity_heatmap_data_merged,
        x='Hour of Day', # Corrected column name to match DataAnalyzerService output
        y='Day of Week', # Corrected column name to match DataAnalyzerService output
        z='Commits',     # Corrected column name to match DataAnalyzerService output
        title='Commit Activity Heatmap (Hour of Day vs. Day of Week)',
        labels={'Hour of Day': 'Hour of Day', 'Day of Week': 'Day of Week', 'Commits': 'Number of Commits'},
        color_continuous_scale="Viridis",
        category_orders={"Day of Week": weekdays} # Ensure correct order
    )
    fig.update_xaxes(side="top", tickvals=list(range(24)))
    fig.update_layout(yaxis_autorange="reversed") # Invert y-axis to have Monday at top
    return fig

def plot_file_extension_changes(file_extension_data):
    """
    Plots the distribution of file extension changes (total lines added/deleted).
    Note: DataAnalyzerService.prepare_file_extension_data now returns 'Extension' and 'Changes' (total lines changed),
    not separate 'added', 'deleted', 'modified' counts. This plot reflects total changes.
    """
    if file_extension_data.empty:
        st.write("No file extension data available.")
        return None

    # This plot is adapted to the output of DataAnalyzerService, which provides 'Extension' and 'Changes'.
    fig = px.bar(
        file_extension_data.head(10), # Display top 10 extensions
        x='Extension',
        y='Changes',
        title='File Extension Total Changes (Top 10)',
        labels={'Extension': 'File Extension', 'Changes': 'Total Lines Changed'},
        color_discrete_sequence=px.colors.qualitative.Plotly # Cycles through colors for each bar
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_file_churn_ranking(file_churn_data):
    """
    Plots a bar chart of the top N files by churn (modification frequency).
    """
    if file_churn_data.empty:
        st.write("No file churn data available.")
        return None

    # Allow user to select the number of files to display
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
        height=min(600, 50 * top_n + 150) # Adjust height based on number of bars for better readability
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort bars by churn count
    return fig

# --- Analysis Orchestration Function ---

# @st.cache_data(show_spinner=False)
def run_analysis(repo_url, repo_path, progress_callback):
    """
    Orchestrates the git mining, data analysis, and AI profiling.
    Uses a progress callback for Streamlit.
    """
    ai_profiler = AIProfilerService() # Initialize AI service
    
    results = {}
    temp_repo_dir = None # To store path of temporary cloned repo

    try:
        current_repo_path = repo_path # Start with local path if provided
        
        # If a GitHub URL is provided and no local path, clone it to a temporary directory.
        if repo_url and not repo_path:
            progress_callback(5, f"Cloning repository from {repo_url}...")
            logger.info(f"Cloning GitHub repository: {repo_url}")
            temp_repo_dir = tempfile.mkdtemp() # Create a temporary directory for cloning
            
            try:
                # Use git command to clone the repository. Limiting depth for faster analysis.
                subprocess.run(['git', 'clone', '--depth', '100', repo_url, temp_repo_dir], check=True)
                current_repo_path = temp_repo_dir
                logger.info(f"Repository cloned to temporary directory: {current_repo_path}")
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to clone repository from URL: {repo_url}. Please ensure the URL is correct and the repository is public or you have access. Error: {e}")
                logger.error(f"Failed to clone repository: {e}", exc_info=True)
                return None
            except Exception as e:
                st.error(f"An unexpected error occurred during cloning: {e}")
                logger.error(f"Cloning failed: {e}", exc_info=True)
                return None
        
        if not current_repo_path:
            st.error("No valid repository path determined for analysis. Please provide a local path or a GitHub URL.")
            return None

        # Step 1: Mine Git Repository
        progress_callback(10, "Mining Git Repository...")
        logger.info(f"Mining repository: Path={current_repo_path}")
        # mine_git_repository now returns a tuple: (unique_commits_df, file_modifications_df)
        unique_commits_df, file_modifications_df = mine_git_repository(
            repo_path=current_repo_path,
            progress_callback=lambda current, total, msg: progress_callback(10 + int(current/total*20), msg) # Adjust progress % for mining
        )
        
        if unique_commits_df.empty:
            st.error("No commit data found for the provided repository. Please check the repository path/URL.")
            return None

        # Initialize DataAnalyzerService with both DataFrames
        analyzer = DataAnalyzerService(unique_commits_df, file_modifications_df)

        # Step 2: Prepare Monthly Commit Data
        progress_callback(35, "Analyzing Monthly Commits...")
        # DataAnalyzerService methods now use internal dataframes, no need to pass 'commits_df'
        monthly_commit_data = analyzer.prepare_monthly_commit_data()
        results['monthly_commit_data'] = monthly_commit_data

        # Step 3: Prepare Activity Heatmap Data
        progress_callback(45, "Analyzing Commit Activity Heatmap...")
        activity_heatmap_data = analyzer.prepare_activity_heatmap_data()
        results['activity_heatmap_data'] = activity_heatmap_data

        # Step 4: Prepare File Extension Data
        progress_callback(60, "Analyzing File Extension Changes...")
        file_extension_data = analyzer.prepare_file_extension_data()
        results['file_extension_data'] = file_extension_data

        # Step 5: Prepare File Churn Ranking Data
        progress_callback(75, "Analyzing File Churn Ranking...")
        file_churn_data = analyzer.prepare_file_churn_ranking_data()
        results['file_churn_data'] = file_churn_data

        # Step 6: Generate Analysis Summary for AI
        progress_callback(85, "Generating Analysis Summary for AI...")
        # DataAnalyzerService.generate_analysis_summary now takes no arguments
        analysis_summary_dict = analyzer.generate_analysis_summary()
        # The AI profiler expects a string prompt, so we extract the narrative summary from the dictionary.
        analysis_summary = analysis_summary_dict['summary_text']
        results['analysis_summary'] = analysis_summary
        
        # Step 7: Get AI Analysis with Retry Logic
        progress_callback(90, "Calling AI for deep analysis (this may take a minute)...")
        logger.info("Calling AI profiler service...")

        # Define retry handler for AI service calls using tenacity
        ai_analysis_with_retry = retry(
            stop=stop_after_attempt(3), # Maximum of 3 attempts
            wait=wait_exponential(multiplier=1, min=2, max=10), # Exponential backoff: 2s, then 4s, up to 10s max
            retry=retry_if_exception_type(Exception), # Retry on any Python Exception
            reraise=True, # Re-raise the last exception if all retries fail
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying AI analysis (attempt {retry_state.attempt_number}/{retry_state.max_attempts_reached + 1})..."
            )
        )(ai_profiler.get_ai_analysis) # Decorate the specific method call

        try:
            ai_summary_results = ai_analysis_with_retry(analysis_summary)
            results['ai_summary_results'] = ai_summary_results
        except Exception as e:
            logger.error(f"Failed to get AI analysis after multiple retries: {e}", exc_info=True)
            st.error(f"Failed to get AI analysis after multiple retries. Please verify your API key and network connection, then try again. Error: {e}")
            results['ai_summary_results'] = "AI analysis failed."

        progress_callback(100, "Analysis complete!")
        return results

    except Exception as e:
        logger.error(f"An unexpected error occurred during analysis: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during analysis: {e}")
        return None
    finally:
        # Clean up the temporary directory if it was created during cloning
        if temp_repo_dir and os.path.exists(temp_repo_dir):
            try:
                shutil.rmtree(temp_repo_dir)
                logger.info(f"Cleaned up temporary cloned repository at: {temp_repo_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary repository {temp_repo_dir}: {e}")

# --- Main Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="Git Repository AI Profiler")

    st.title("Git Repository AI Profiler")

    # Sidebar for repository input
    with st.sidebar:
        st.header("Repository Input")
        repo_option = st.radio(
            "Select repository source:",
            ("Upload Git Bundle (.bundle)", "Enter Local Path", "Enter GitHub URL")
        )

        repo_path_for_analysis = None
        repo_url_for_analysis = None

        if repo_option == "Upload Git Bundle (.bundle)":
            uploaded_file = st.file_uploader("Upload .bundle file", type="bundle")
            if uploaded_file is not None:
                st.warning("Handling '.bundle' files requires Git command-line tools for unbundling, which are currently restricted. "
                           "Please use 'Enter Local Path' after manually cloning/unbundling the bundle, or 'Enter GitHub URL'.")
                # Ensure no analysis is triggered if bundle is uploaded but cannot be processed
                repo_path_for_analysis = None 
                repo_url_for_analysis = None

        elif repo_option == "Enter Local Path":
            local_path_input = st.text_input("Enter path to local Git repository (e.g., /Users/user/my_repo)", value=os.getcwd())
            if os.path.isdir(local_path_input) and os.path.exists(os.path.join(local_path_input, '.git')):
                repo_path_for_analysis = local_path_input
                st.success("Valid local Git repository found.")
            elif local_path_input:
                st.warning("The provided path is not a valid Git repository or does not exist.")

        elif repo_option == "Enter GitHub URL":
            github_url = st.text_input("Enter GitHub repository URL (e.g., https://github.com/streamlit/streamlit)", value="https://github.com/streamlit/streamlit")
            if github_url:
                repo_url_for_analysis = github_url
                st.success("GitHub URL provided. It will be cloned temporarily for analysis.")

        st.subheader("Analysis Controls")
        # Enable button only if a valid path OR a URL is provided
        if st.button("Analyze Repository", type="primary", disabled=not (bool(repo_path_for_analysis) or bool(repo_url_for_analysis))):
            st.session_state['run_analysis'] = True
            st.session_state['repo_path_param'] = repo_path_for_analysis
            st.session_state['repo_url_param'] = repo_url_for_analysis
            st.session_state['analysis_results'] = None # Clear previous results
        
        # Add a clear cache button for development/debugging
        if st.button("Clear Cache & Reset"):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()

    # --- Main Content Area ---
    if 'run_analysis' in st.session_state and st.session_state['run_analysis']:
        if st.session_state.get('repo_path_param') or st.session_state.get('repo_url_param'):
            display_repo_info = st.session_state.get('repo_url_param') or st.session_state.get('repo_path_param')
            st.info(f"Starting analysis for repository: {display_repo_info}")
            
            # Placeholder for progress bar and text
            progress_text_placeholder = st.empty()
            progress_bar_placeholder = st.progress(0)

            def update_progress_ui(percent_complete, message):
                progress_bar_placeholder.progress(int(percent_complete) / 100)
                progress_text_placeholder.text(f"Progress: {message} ({int(percent_complete)}%)")

            # Run analysis and store results in session state
            results = run_analysis(
                repo_url=st.session_state.get('repo_url_param'),
                repo_path=st.session_state.get('repo_path_param'),
                progress_callback=update_progress_ui
            )
            st.session_state['analysis_results'] = results
            st.session_state['run_analysis'] = False # Reset analysis flag

            # The temporary cleanup logic for 'base_temp_dir_for_cleanup' was for old logic.
            # The current temporary repo cleanup is handled in the `finally` block of `run_analysis`.
            # This block can be removed to avoid confusion.
            # if 'base_temp_dir_for_cleanup' in st.session_state and os.path.exists(st.session_state['base_temp_dir_for_cleanup']):
            #     try:
            #         shutil.rmtree(st.session_state['base_temp_dir_for_cleanup'])
            #         logger.info(f"Cleaned up temporary directory: {st.session_state['base_temp_dir_for_cleanup']}")
            #         del st.session_state['base_temp_dir_for_cleanup']
            #     except Exception as e:
            #         logger.error(f"Error cleaning up old temporary directory {st.session_state['base_temp_dir_for_cleanup']}: {e}")
            
            st.rerun() # Rerun to display results
        else:
            st.error("No repository source selected or valid path/URL provided for analysis.")

    if 'analysis_results' in st.session_state and st.session_state['analysis_results'] is not None:
        results = st.session_state['analysis_results']

        st.success("Analysis Complete!")

        # Display AI Analysis Summary
        st.subheader("AI-Powered Repository Insights")
        if 'ai_summary_results' in results and results['ai_summary_results'] and results['ai_summary_results'] != "AI analysis failed.":
            st.markdown(results['ai_summary_results'])
        else:
            st.warning("AI analysis results are not available or an error occurred.")

        st.subheader("Detailed Repository Metrics")

        # Create tabs for different visualizations
        tab_titles = ["Monthly Commits", "Activity Heatmap", "File Extension Changes", "File Churn Ranking"]
        tabs = st.tabs(tab_titles)

        with tabs[0]:
            st.header("Monthly Commit Activity")
            if 'monthly_commit_data' in results and not results['monthly_commit_data'].empty:
                st.plotly_chart(plot_monthly_commits(results['monthly_commit_data']), use_container_width=True)
            else:
                st.info("No monthly commit data to display.")

        with tabs[1]:
            st.header("Commit Activity Heatmap")
            if 'activity_heatmap_data' in results and not results['activity_heatmap_data'].empty:
                st.plotly_chart(plot_activity_heatmap(results['activity_heatmap_data']), use_container_width=True)
            else:
                st.info("No activity heatmap data to display.")

        with tabs[2]:
            st.header("File Extension Change Distribution")
            if 'file_extension_data' in results and not results['file_extension_data'].empty:
                st.plotly_chart(plot_file_extension_changes(results['file_extension_data']), use_container_width=True)
            else:
                st.info("No file extension change data to display.")

        with tabs[3]: # New Tab for File Churn Ranking
            st.header("File Churn Ranking")
            st.write("This chart highlights files that are frequently modified across commits, which can indicate areas of high development activity, potential complexity, or instability in the codebase.")
            if 'file_churn_data' in results and not results['file_churn_data'].empty:
                st.plotly_chart(plot_file_churn_ranking(results['file_churn_data']), use_container_width=True)
            else:
                st.info("No file churn ranking data to display.")

    else:
        st.info("Upload or provide a Git repository path/URL and click 'Analyze Repository' to start.")
        # Only show "Ready to analyze" if a valid repo source is selected but analysis hasn't run yet
        if not st.session_state.get('run_analysis', False) and (st.session_state.get('repo_path_param') or st.session_state.get('repo_url_param')):
            display_repo_info = st.session_state.get('repo_url_param') or st.session_state.get('repo_path_param')
            st.success(f"Ready to analyze: {display_repo_info}")


if __name__ == "__main__":
    main()