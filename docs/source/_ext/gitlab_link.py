"This is copied from https://github.com/scikit-learn/scikit-learn/blob/main/doc/sphinxext/github_link.py"

import inspect
import os
import subprocess
import sys
from operator import attrgetter

REVISION_CMD = "git rev-parse HEAD"


def _get_git_revision() -> None | str:
    try:
        revision = subprocess.check_output(REVISION_CMD.split()).strip()
    except (subprocess.CalledProcessError, OSError):
        print("Failed to execute git to get revision")
        return None
    return revision.decode("utf-8")


def linkcode_resolve(domain, info) -> None | str:
    """Determine a link to online source for a class/method/function

    This is called by sphinx.ext.linkcode

    An example with a long-untouched module that everyone has
    >>> _linkcode_resolve('py', {'module': 'tty',
    ...                          'fullname': 'setraw'},
    ...                   package='tty',
    ...                   url_fmt='https://hg.python.org/cpython/file/'
    ...                           '{revision}/Lib/{package}/{path}#L{lineno}',
    ...                   revision='xxxx')
    'https://hg.python.org/cpython/file/xxxx/Lib/tty/tty.py#L18'
    """
    package = "idtrackerai"
    url_fmt = "https://gitlab.com/polavieja_lab/idtrackerai/blob/{revision}/src/{package}/{path}#L{lineno}"
    revision = _get_git_revision()

    if revision is None:
        return
    if domain not in ("py", "pyx"):
        return
    if not info.get("module") or not info.get("fullname"):
        return

    obj = None
    try:
        class_name = info["fullname"].split(".")[0]
        module = __import__(info["module"], fromlist=[class_name])
        obj = attrgetter(info["fullname"])(module)

        # Unwrap the object to get the correct source
        # file in case that is wrapped by a decorator
        obj = inspect.unwrap(obj)

        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None

    if obj is None:
        return None

    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return

    fn = os.path.relpath(fn, start=os.path.dirname(__import__(package).__file__))
    try:
        lineno = inspect.getsourcelines(obj)[1]
    except Exception:
        lineno = ""
    return url_fmt.format(revision=revision, package=package, path=fn, lineno=lineno)
