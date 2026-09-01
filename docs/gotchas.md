# Repository Gotchas

## 2026-09-01: Configured checkout path was missing

The configured workspace path did not exist when repository work resumed.
A separate checkout existed under `Desktop/dev/GitHubNew`, but it contained untracked local environment and history files.
The verified fix was to create a clean worktree at the configured workspace path from `origin/main` and leave the separate checkout untouched.

## 2026-09-01: GitHub CLI authentication was expired

The GitHub CLI reported that the active account token was invalid.
Public repository data remained available through Git, so branch review and local verification could continue.
Publishing changes requires either working Git credentials or re-authentication with `gh auth login -h github.com`.

## 2026-09-01: Local smoke tests initially lacked dependencies

The restored worktree used the system Python installation, which did not have NumPy or PyTorch installed.
The smoke-test failures were import failures rather than failures in repository behavior.
The repository CI installs all declared dependencies under Python 3.11 before running the same smoke tests.

## 2026-09-01: Historical commits contain automated co-author trailers

Several commits already merged through pull requests 1 and 2 contain automated model co-author trailers.
Rewriting published history would be destructive, so those commits were left intact.
New commits must not add automated agent or model co-authors.
