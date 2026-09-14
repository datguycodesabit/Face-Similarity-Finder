# Publish with GitHub Desktop

The repository to add is the **outer project folder**:

```text
/Users/carmelodalio/Documents/ChatGPT/Face Data Point Similarity Finder
```

Do not select the smaller `Face-Similarity-Finder` folder inside it. That is an old clone and is ignored by the real project.

## First upload

1. Open GitHub Desktop.
2. Choose **File → Add Local Repository**.
3. Select `Documents/ChatGPT/Face Data Point Similarity Finder`.
4. Review the Changes tab. It should contain source, tests, documentation, and—after curation—the Commons metadata manifest only; it must not contain `data`, `models`, virtual environments, browser downloads, databases, MICA/FLAME assets, or portrait files.
5. Enter a summary such as `Add three-view face shape studio` and choose **Commit to main**.
6. Choose **Push origin**.

The repository is connected to:

```text
https://github.com/datguycodesabit/Face-Similarity-Finder
```

## What GitHub intentionally will not contain

- The legacy LFW dataset and all downloaded Commons photographs
- The separately licensed MICA checkout, MICA weights, FLAME model, and MICA environment
- `data/face_match.sqlite3`
- `models/face_landmarker.task`
- Python and Node dependency folders
- Playwright browsers and generated screenshots
- Local caches and temporary test fixtures

These are regenerated using the setup commands in `README.md`. Excluding them keeps the repository small, avoids GitHub's file-size limits, and prevents redistribution of portraits or separately licensed model assets.

## Future updates

Open this repository in GitHub Desktop, review the changed source files, write a commit summary, choose **Commit to main**, and then **Push origin**.
