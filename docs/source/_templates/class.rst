{{ objname | escape | underline(line="=") }}

.. warning::

   The code reference is a work in progress and may contain inconsistencies.

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
