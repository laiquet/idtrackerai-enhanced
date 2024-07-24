{{ objname | escape | underline(line="=") }}

.. currentmodule:: {{ module }}

{% if objtype == "module" -%}

.. automodule:: {{ fullname }}
   :members:

{%- elif objtype == "function" -%}

.. autofunction:: {{ objname }}

{%- elif objtype == "class" -%}

.. autoclass:: {{ objname }}
   :members:

{%- else -%}

.. auto{{ objtype }}:: {{ objname }}

{%- endif -%}
