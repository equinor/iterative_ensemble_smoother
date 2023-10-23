..
   Template for the html base rendering

   Modified from
   https://github.com/sphinx-doc/sphinx/tree/master/sphinx/ext/autosummary/templates/autosummary/base.rst

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
