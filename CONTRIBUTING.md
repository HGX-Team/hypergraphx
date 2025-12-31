# Contributing to Hypergraphx on GitHub

Follow this step-by-step guide to contribute to Hypergraphx.

## 1. Set Up Your GitHub Account

If you haven't already, create a GitHub account at [GitHub.com](https://github.com/).

## 2. Fork the Repository

- Go to the main page of the Hypergraphx repository.
- In the top-right corner of the page, click on the "Fork" button. This will create a copy of the repository in your
  GitHub account.

## 3. Clone Your Forked Repository

- Navigate to your forked repository in your GitHub account.
- Click the "Code" button and copy the URL.
- Open your terminal and navigate to the directory where you want to clone the repository.
- Run the following command:

```bash
git clone [URL]
```

Replace `[URL]` with the URL you copied.

## 4. Set Upstream Remote

To keep your forked repository updated with the changes from the original repository, you need to set an upstream
remote:

- Navigate to the directory of your cloned repository in the terminal.
- Run the following command:

```bash
git remote add upstream https://github.com/HGX-Team/hypergraphx.git
```

## 5. Create a New Branch

Before making any changes, it's a good practice to create a new branch:

- Navigate to the directory of your cloned repository in the terminal.
- Run the following command to create and switch to a new branch:

```bash
git checkout -b your-branch-name
```

## 6. Make Your Changes

- Edit the files or add new files as required.
- Once you've made your changes, save them.
- Format Python code with Black before committing:

```bash
black .
```

- (Recommended) Install pre-commit so formatting happens automatically:

```bash
pre-commit install
```

## 7. Commit Your Changes

- In the terminal, navigate to the directory of your cloned repository.
- Run the following commands to add and commit your changes:

```bash
git add .
git commit -m "Your commit message here"
```

## 8. Push Your Changes to GitHub

- Push your changes to your forked repository on GitHub:

```bash
git push origin your-branch-name
```

## 9. Create a Pull Request (PR)

- Go to your forked repository on GitHub.
- Click on the "Pull requests" tab and then click on the "New pull request" button.
- Ensure the base repository is the original Hypergraphx repository and the base branch is the branch you want to merge
  your changes into (usually `main`).
- Ensure the head repository is your forked repository and the compare branch is the branch you made your changes in.
- Click on the "Create pull request" button.
- Fill in the PR title and description, explaining your changes.
- Click on the "Create pull request" button to submit your PR.

## 10. Wait for Review

- The maintainers of the Hypergraphx repository will review your PR.
- They might request some changes or improvements. If so, make the required changes in your branch, commit them, and
  push them to GitHub. Your PR will be automatically updated.

## 11. PR Gets Merged

Once your PR is approved, the maintainers will merge it into the main branch of the Hypergraphx repository.

**Note:** Always follow the contribution guidelines provided by the repository maintainers, and always be respectful and
constructive in your interactions.
