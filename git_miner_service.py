import os
from pydriller import Repository, Git
import pandas as pd
from datetime import datetime

def mine_git_repository(repo_path: str, progress_callback=None) -> (pd.DataFrame, pd.DataFrame):
    """
    Mines a Git repository to extract commit and file change data.

    This function iterates through all commits in a specified Git repository,
    collecting detailed information about each commit and the files modified
    within them. It can optionally report progress via a callback function,
    which is useful for UI updates (e.g., Streamlit progress bars).

    Args:
        repo_path (str): The absolute or relative path to the Git repository.
                         This path should point to the root directory of the repository.
        progress_callback (callable, optional): A function to call with progress updates.
                                                If provided, it should accept three arguments:
                                                (current_step: int, total_steps: int, message: str).
                                                Defaults to None, meaning no progress reporting.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
               - commits_df (pd.DataFrame): DataFrame of commit-level data, with columns
                                            like 'hash', 'author_name', 'author_date', 'message',
                                            'lines_added_commit', 'lines_deleted_commit',
                                            'files_changed_commit'.
               - file_changes_df (pd.DataFrame): DataFrame of file-level change data, with columns
                                                 like 'commit_hash', 'change_type', 'old_path',
                                                 'new_path', 'file_path', 'lines_added',
                                                 'lines_deleted', 'nloc', 'complexity'.

    Raises:
        FileNotFoundError: If the specified repository path does not exist.
        ValueError: If the specified path is not a valid Git repository.
    """
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    if not os.path.isdir(os.path.join(repo_path, '.git')):
        raise ValueError(f"'{repo_path}' is not a valid Git repository.")

    commits_data = []
    file_changes_data = []

    # Attempt to get the total number of commits for an accurate progress bar
    try:
        git_helper = Git(repo_path)
        total_commits = git_helper.total_commits()
    except Exception as e:
        # Fallback if `Git().total_commits()` fails (e.g., repository corrupted, no commits)
        print(f"Warning: Could not get total commit count for repository {repo_path}: {e}")
        total_commits = 0  # Indicate unknown total

    current_commit_count = 0

    # Initialize Repository miner
    repo_miner = Repository(repo_path)

    for commit in repo_miner.traverse_commits():
        current_commit_count += 1

        # Report progress if a callback is provided
        if progress_callback:
            # If total_commits is unknown (0), we estimate total to prevent ZeroDivisionError
            # and allow the progress bar to show relative movement.
            display_total = total_commits if total_commits > 0 else current_commit_count + 1
            progress_callback(
                current_commit_count,
                display_total,
                f"Mining commit {commit.hash[:7]} by {commit.author.name}"
            )

        # Collect commit-level data
        commits_data.append({
            'hash': commit.hash,
            'author_name': commit.author.name,
            'author_email': commit.author.email,
            'author_date': commit.author_date,
            'committer_name': commit.committer.name,
            'committer_email': commit.committer.email,
            'committer_date': commit.committer_date,
            'message': commit.msg,
            'lines_added_commit': commit.insertions,
            'lines_deleted_commit': commit.deletions,
            'files_changed_commit': len(commit.modified_files)
        })

        # Collect file-level change data for each modification in the commit
        for mod in commit.modified_files:
            file_changes_data.append({
                'commit_hash': commit.hash,
                'change_type': mod.change_type.name,  # e.g., ADD, DELETE, MODIFY, RENAME, COPY
                'old_path': mod.old_path,
                'new_path': mod.new_path,
                # 'file_path' represents the path of the file *after* the change.
                # For deleted files, new_path is None, so old_path is used.
                'file_path': mod.new_path if mod.new_path else mod.old_path,
                'lines_added': mod.added_lines,
                'lines_deleted': mod.deleted_lines,
                # Pydriller can return None for nloc/complexity, default to 0 for int compatibility
                'nloc': mod.nloc if mod.nloc is not None else 0,
                'complexity': mod.complexity if mod.complexity is not None else 0
            })

    # Convert collected data into pandas DataFrames
    commits_df = pd.DataFrame(commits_data)
    file_changes_df = pd.DataFrame(file_changes_data)

    # Post-processing for commits_df
    if not commits_df.empty:
        # Ensure author_date and committer_date are timezone-aware datetime objects, converted to UTC
        commits_df['author_date'] = pd.to_datetime(commits_df['author_date'], utc=True)
        commits_df['committer_date'] = pd.to_datetime(commits_df['committer_date'], utc=True)
    else:
        # Define empty DataFrame with correct dtypes if no commits were found
        commits_df = pd.DataFrame(columns=[
            'hash', 'author_name', 'author_email', 'author_date',
            'committer_name', 'committer_email', 'committer_date', 'message',
            'lines_added_commit', 'lines_deleted_commit', 'files_changed_commit'
        ]).astype({
            'hash': str, 'author_name': str, 'author_email': str,
            'author_date': 'datetime64[ns, UTC]', 'committer_name': str,
            'committer_email': str, 'committer_date': 'datetime64[ns, UTC]',
            'message': str, 'lines_added_commit': int,
            'lines_deleted_commit': int, 'files_changed_commit': int
        })

    # Post-processing for file_changes_df
    if not file_changes_df.empty:
        # Fill potential None values in path columns with empty strings for consistency
        file_changes_df['old_path'] = file_changes_df['old_path'].fillna('').astype(str)
        file_changes_df['new_path'] = file_changes_df['new_path'].fillna('').astype(str)
        file_changes_df['file_path'] = file_changes_df['file_path'].fillna('').astype(str)
    else:
        # Define empty DataFrame with correct dtypes if no file changes were found
        file_changes_df = pd.DataFrame(columns=[
            'commit_hash', 'change_type', 'old_path', 'new_path', 'file_path',
            'lines_added', 'lines_deleted', 'nloc', 'complexity'
        ]).astype({
            'commit_hash': str, 'change_type': str, 'old_path': str,
            'new_path': str, 'file_path': str, 'lines_added': int,
            'lines_deleted': int, 'nloc': int, 'complexity': int
        })

    return commits_df, file_changes_df