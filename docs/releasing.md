# Release process

PanelWise uses semantic versions and publishes from signed Git tags through GitHub Actions.

## Before tagging

1. Confirm `main` is green and the worktree is clean.
2. Update `src/panelwise/_version.py`, `pyproject.toml`, and `CHANGELOG.md` to the same version.
3. Run:

   ```bash
   python -m unittest discover -s tests -v
   python -m build
   python -m pip install --force-reinstall dist/panelwise-*.whl
   panelwise --version
   ```

4. Inspect wheel contents and run the examples with a non-production key.
5. Open and merge a focused release PR.

## Publish

```bash
git tag -s v0.1.0 -m "PanelWise 0.1.0"
git push origin v0.1.0
```

The release workflow verifies that the tag matches the package version, runs tests, builds distributions, uploads artifacts to the GitHub release, and publishes to PyPI through Trusted Publishing. Maintainers must configure the `pypi` GitHub environment before the first public release.

Never publish from an unreviewed local worktree or embed an API token in the workflow.
