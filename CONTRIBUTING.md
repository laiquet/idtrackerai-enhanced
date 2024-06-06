# Contributing to idtracker.ai

First off, thanks for taking the time to contribute! The following is a set of guidelines for contributing to idtracker.ai. These are mostly guidelines, not rules, use your best judgment 🙂.

If you are not used to work with open-source projects or you don't feel comfortable with git or GitLab, you can communicate your ideas/issues in the [idtracker.ai users group](https://groups.google.com/forum/#!forum/idtrackerai_users) or by email at [idtrackerai@gmail.com](mailto:idtrackerai@gmail.com).

## Code of Conduct

This project, and anyone participating in it, is governed by the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## Issues

Bugs, feature requests and any other technical comments about the software are managed by using GitLab Issues. Before opening one:

* Check if you are running the latest version of idtracker.ai (``python -m pip install -U idtrackerai``).
* Check the [FAQs on the webpage](https://idtracker.ai/latest/user_guide/FAQs.html), the [installation troubleshooting guide](https://idtracker.ai/latest/install/installation_troubleshooting.html), and the [idtracker.ai users group](https://groups.google.com/forum/#!forum/idtrackerai_users) for a list of common questions and problems.
* Perform a [cursory search](https://gitlab.com/polavieja_lab/idtrackerai/issues) to see if the problem has already been reported. If it has **and the issue is still open**, add a comment to the existing issue instead of opening a new one.

While writing your issue:

* Explain the problem and include additional details to help maintainers reproduce the problem.
* Attach the entire log file to the issue or add the text in your terminal if it contains some extra message not present in the log file.

## Repo and code structure

The repository has two main branches, ``master`` and ``v5-dev``. No commits should never be pushed to ``master``, this branch should only be modified by merge requests from ``v5-dev``. When one of these merge happens, a new version of idtrackerai should be released by creating a tag on the ``master`` branch (then, GitLab will automatically run a [CICD pipeline](.gitlab-ci.yml) to publish the branch to [PyPI](https://pypi.org/project/idtrackerai/)). ``v5-dev`` is the developing branch where all commits are accumulated until they are all merged into ``master``. Other branches are created for specific new feature implementations or for testing and their final goal must be to be merged into the developing branch and be removed afterwards.

All commits must pass the [Pre-commit](https://pre-commit.com/) check which follows this [config file](.pre-commit-config.yaml). It mainly consists of [Black](https://black.readthedocs.io/), [isort](https://pycqa.github.io/isort/) and [flake8](https://flake8.pycqa.org/). To use pre-commit:

```
pip install pre-commit # get the pre-commit package
pre-commit install # install pre-commit in the repo so that git knows about its existence
```

And done! Next commits will be automatically checked with the hooks indicated in the [config file](.pre-commit-config.yaml).

The [pyproject.toml](pyproject.toml) contains the package information and building metadata as well as other settings for developing.

All tests must succeed, you can run them with [PyTest](https://docs.pytest.org/) using the command ``pytest .``. New code should include tests for it in the [test folder](tests).

Relevant code modifications should always include a record in [the changelog](docs/source/user_guide/changelog.rst).

Happy coding!
