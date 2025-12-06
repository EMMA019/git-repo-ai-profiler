import pandas as pd
from datetime import datetime
import os

# Helper function (private to this module)
def _get_file_extension(filepath: str) -> str:
    """Extracts the file extension from a given file path.
    Handles None/NaN paths and returns 'no_extension' for consistency.
    Also handles files like .gitignore correctly.
    """
    if pd.isna(filepath) or not isinstance(filepath, str):
        return 'no_extension'
    # Use os.path.splitext, then ensure it's lower case.
    # If no extension (e.g., 'file' or '.gitignore'), ext will be empty or '.gitignore' itself.
    base, ext = os.path.splitext(filepath)
    if not ext and base and base.startswith('.'): # Handles files like '.gitignore' or '.env'
        return base.lower()
    return ext.lower() if ext else 'no_extension'

class DataAnalyzerService:
    """
    Service class responsible for processing raw Git commit data
    into aggregated and summarized formats suitable for visualization
    and AI analysis.
    """

    def __init__(self, unique_commits_df: pd.DataFrame, file_modifications_df: pd.DataFrame):
        """
        Initializes the DataAnalyzerService with the raw commit and file modification data.

        Args:
            unique_commits_df: DataFrame with unique commit-level data from git_miner_service.
            file_modifications_df: DataFrame with file-level modification data from git_miner_service.
        """
        # Ensure unique_commits_df is not empty before processing
        if unique_commits_df.empty:
            self.unique_commits_df = pd.DataFrame(columns=[
                'hash', 'author_date', 'author_name', 'insertions', 'deletions',
                'lines_added_commit', 'lines_deleted_commit' # Ensure these columns are present for summary
            ])
        else:
            self.unique_commits_df = unique_commits_df.copy()
            # Ensure author_date is datetime. Convert timezone-aware to timezone-naive for consistent calculations.
            # This makes dt.weekday, dt.hour, to_period('M') behave predictably without timezone complications.
            self.unique_commits_df['author_date'] = pd.to_datetime(self.unique_commits_df['author_date']).dt.tz_localize(None)

        # Ensure file_modifications_df is not empty before processing
        if file_modifications_df.empty:
            self.file_modifications_df = pd.DataFrame(columns=[
                'commit_hash', 'change_type', 'file_path', 'lines_added', 'lines_deleted', 'extension'
            ])
        else:
            self.file_modifications_df = file_modifications_df.copy()
            # Apply robust file extension extraction using the internal helper for consistency
            self.file_modifications_df['extension'] = self.file_modifications_df['file_path'].apply(_get_file_extension)


    def prepare_monthly_commit_data(self) -> pd.DataFrame:
        """
        Prepares data for monthly commit count trend visualization, counting unique commits.

        Returns:
            DataFrame with 'Month' (datetime) and 'Commits' (count).
        """
        if self.unique_commits_df.empty:
            return pd.DataFrame(columns=['Month', 'Commits'])
        
        # Group by year and month from unique commits and count
        monthly_commits = self.unique_commits_df.groupby(
            self.unique_commits_df['author_date'].dt.to_period('M')
        ).size().reset_index(name='Commits')
        
        # Convert Period to datetime for easier plotting with Plotly
        monthly_commits['Month'] = monthly_commits['author_date'].dt.to_timestamp()
        
        return monthly_commits[['Month', 'Commits']].sort_values('Month')

    def prepare_activity_heatmap_data(self) -> pd.DataFrame:
        """
        Prepares data for activity heatmap (Day of Week vs. Hour of Day), counting unique commits.

        Returns:
            DataFrame with 'Hour of Day', 'Day of Week', 'Commits'. This is in 'long' format.
        """
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        if self.unique_commits_df.empty:
            # Return an empty heatmap structure consistent with plot_activity_heatmap's expectation (long format)
            return pd.DataFrame(columns=['Hour of Day', 'Day of Week', 'Commits'])

        # Extract weekday (0=Monday, 6=Sunday) and hour from unique commits
        self.unique_commits_df['weekday'] = self.unique_commits_df['author_date'].dt.weekday
        self.unique_commits_df['hour'] = self.unique_commits_df['author_date'].dt.hour

        # Group by weekday and hour, then count unique commits
        activity_counts = self.unique_commits_df.groupby(['weekday', 'hour']).size().reset_index(name='Commits')

        # Create a full grid to ensure all hours and weekdays are represented, filling missing with 0
        all_hours = pd.RangeIndex(start=0, stop=24)
        all_weekdays_nums = pd.RangeIndex(start=0, stop=7)
        full_grid = pd.MultiIndex.from_product([all_weekdays_nums, all_hours], names=['weekday', 'hour']).to_frame(index=False)
        
        # Merge with actual activity counts
        heatmap_df = pd.merge(full_grid, activity_counts, on=['weekday', 'hour'], how='left').fillna(0)
        
        # Map weekday numbers to names and rename hour column
        heatmap_df['Day of Week'] = heatmap_df['weekday'].map(lambda x: weekday_names[x])
        heatmap_df['Hour of Day'] = heatmap_df['hour']

        return heatmap_df[['Hour of Day', 'Day of Week', 'Commits']].astype({'Commits': int})


    def prepare_file_extension_data(self) -> pd.DataFrame:
        """
        Prepares data for file extension changes visualization.
        Counts total lines changed (added + deleted) per file extension.

        Returns:
            DataFrame with 'Extension', 'Changes' (total lines changed).
        """
        if self.file_modifications_df.empty:
            return pd.DataFrame(columns=['Extension', 'Changes'])
        
        # The file_modifications_df already has 'extension', 'lines_added', 'lines_deleted' for each file modification
        # Sum file-level added and deleted lines for each modification
        # Ensure columns exist and are numeric, default to 0 if not present or non-numeric
        lines_added = pd.to_numeric(self.file_modifications_df.get('lines_added', 0), errors='coerce').fillna(0)
        lines_deleted = pd.to_numeric(self.file_modifications_df.get('lines_deleted', 0), errors='coerce').fillna(0)
        
        self.file_modifications_df['file_changes_lines'] = lines_added + lines_deleted
        
        # Group by extension and sum the changes
        extension_summary = self.file_modifications_df.groupby('extension')['file_changes_lines'].sum().reset_index(name='Changes')

        # Rename columns for clarity in plots
        extension_summary.rename(columns={'extension': 'Extension'}, inplace=True)
        
        # Sort by changes
        extension_summary = extension_summary.sort_values(by='Changes', ascending=False)

        return extension_summary

    def prepare_file_churn_ranking_data(self) -> pd.DataFrame:
        """
        Calculates file churn, ranking files by the number of unique commits they appear in.

        Returns:
            DataFrame with 'file_path' and 'churn_count'.
        """
        if self.file_modifications_df.empty:
            return pd.DataFrame(columns=['file_path', 'churn_count'])

        # Group by file_path and count unique commit_hashes for each file
        file_churn = self.file_modifications_df.groupby('file_path')['commit_hash'].nunique().reset_index(name='churn_count')
        
        # Sort by churn count in descending order
        file_churn = file_churn.sort_values(by='churn_count', ascending=False)

        return file_churn


    def generate_analysis_summary(self) -> dict:
        """
        Generates a summary dictionary of key project metrics for Gemini analysis.

        Returns:
            A dictionary containing key summary statistics and a narrative summary text.
        """
        if self.unique_commits_df.empty:
            return {
                "total_commits": 0,
                "project_duration_days": 0,
                "first_commit_date": "N/A",
                "last_commit_date": "N/A",
                "total_authors": 0,
                "top_author": "N/A",
                "top_author_commits": 0,
                "avg_commits_per_day": 0.0,
                "total_lines_added": 0,
                "total_lines_deleted": 0,
                "most_active_weekday": "N/A",
                "most_active_hour": "N/A",
                "dominant_file_extensions": [],
                "top_churned_files": [],
                "summary_text": "No commit data available for analysis. Please provide a valid Git repository URL with commits."
            }

        total_commits = len(self.unique_commits_df)
        first_commit_date = self.unique_commits_df['author_date'].min()
        last_commit_date = self.unique_commits_df['author_date'].max()
        project_duration_days = (last_commit_date - first_commit_date).days if total_commits > 1 else 0

        total_authors = self.unique_commits_df['author_name'].nunique()
        author_counts = self.unique_commits_df['author_name'].value_counts()
        top_author = author_counts.index[0] if not author_counts.empty else "N/A"
        top_author_commits = int(author_counts.iloc[0]) if not author_counts.empty else 0

        avg_commits_per_day = total_commits / (project_duration_days + 1) if project_duration_days >= 0 else 0
        avg_commits_per_day = round(avg_commits_per_day, 2)

        # Use commit-level 'lines_added_commit' and 'lines_deleted_commit' for total lines changed
        # Ensure columns exist and are numeric, default to 0 if not present or non-numeric
        total_lines_added = pd.to_numeric(self.unique_commits_df.get('lines_added_commit', 0), errors='coerce').fillna(0).sum()
        total_lines_deleted = pd.to_numeric(self.unique_commits_df.get('lines_deleted_commit', 0), errors='coerce').fillna(0).sum()


        # Get activity heatmap data to find most active weekday/hour
        heatmap_data = self.prepare_activity_heatmap_data() # Call internal method
        most_active_weekday = "N/A"
        most_active_hour = "N/A"
        if not heatmap_data.empty and heatmap_data['Commits'].sum() > 0:
            # Find the row with maximum 'Commits'
            max_activity_row = heatmap_data.loc[heatmap_data['Commits'].idxmax()]
            most_active_weekday = max_activity_row['Day of Week']
            most_active_hour = int(max_activity_row['Hour of Day'])


        # Get file extension data
        file_extension_summary = self.prepare_file_extension_data() # Call internal method
        dominant_file_extensions = file_extension_summary.head(3)['Extension'].tolist() if not file_extension_summary.empty else []

        # Get file churn data
        file_churn_summary = self.prepare_file_churn_ranking_data()
        top_churned_files = file_churn_summary.head(3)['file_path'].tolist() if not file_churn_summary.empty else []


        # Construct summary text
        summary_lines = []
        summary_lines.append(f"This repository contains {total_commits} unique commits.")
        if total_commits > 0:
            summary_lines.append(f"It spans {project_duration_days} days, from {first_commit_date.strftime('%Y-%m-%d')} to {last_commit_date.strftime('%Y-%m-%d')}.")
            summary_lines.append(f"There are {total_authors} unique authors. The most active author is '{top_author}' with {top_author_commits} commits.")
            summary_lines.append(f"On average, {avg_commits_per_day} commits are made per day.")
            summary_lines.append(f"A total of {int(total_lines_added)} lines were added and {int(total_lines_deleted)} lines were deleted across all unique commits.")
            if most_active_weekday != "N/A":
                summary_lines.append(f"The peak activity time is typically on {most_active_weekday} around {most_active_hour}:00.")
            if dominant_file_extensions:
                summary_lines.append(f"Dominant file types changed include: {', '.join(dominant_file_extensions)}.")
            else:
                summary_lines.append("No specific dominant file extensions were identified.")
            if top_churned_files:
                summary_lines.append(f"Top churned files (frequently changed) include: {', '.join(top_churned_files)}.")
            else:
                summary_lines.append("No specific churned files were identified.")
        
        summary_text = " ".join(summary_lines)

        summary_data = {
            "total_commits": total_commits,
            "project_duration_days": project_duration_days,
            "first_commit_date": str(first_commit_date.strftime('%Y-%m-%d %H:%M:%S')), # Format for consistent JSON
            "last_commit_date": str(last_commit_date.strftime('%Y-%m-%d %H:%M:%S')),   # Format for consistent JSON
            "total_authors": total_authors,
            "top_author": top_author,
            "top_author_commits": top_author_commits,
            "avg_commits_per_day": avg_commits_per_day,
            "total_lines_added": int(total_lines_added), # Ensure int for JSON serialization
            "total_lines_deleted": int(total_lines_deleted), # Ensure int for JSON serialization
            "most_active_weekday": most_active_weekday,
            "most_active_hour": most_active_hour,
            "dominant_file_extensions": dominant_file_extensions,
            "top_churned_files": top_churned_files, # Added for completeness in summary data
            "summary_text": summary_text
        }
        return summary_data