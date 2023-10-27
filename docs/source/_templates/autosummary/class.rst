..
   Template for the html class rendering

   Modified from
   https://github.com/sphinx-doc/sphinx/tree/master/sphinx/ext/autosummary/templates/autosummary/class.rst

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :private-members:
   :show-inheritance:
   :inherited-members:

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

   .. rubric:: {{ _('Methods definition') }}
