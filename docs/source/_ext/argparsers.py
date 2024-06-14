import re

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util.nodes import nested_parse_with_titles

from idtrackerai.start.arg_parser import get_argparser_help


class IdtrackeraiArgparser(Directive):
    def run(self):
        text = get_argparser_help()

        text = (
            re.sub(r"> \[<\w+> \.{3}\]", " ...>", text)
            .replace("> <", " ")
            .replace("[<path> ...]", "<path ...>")
        )

        lines = text.splitlines()
        options_lines = lines[lines.index("options:") : -1]

        rst = StringList()

        for line in options_lines:
            rst.append(line, "argparser")

        node = nodes.section()
        nested_parse_with_titles(self.state, rst, node)

        return node.children


def setup(app: Sphinx):
    app.add_directive("idtrackerai_argparser", IdtrackeraiArgparser)
    return {"parallel_read_safe": True}
