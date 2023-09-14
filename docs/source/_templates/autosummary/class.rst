{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:                                    <-- add at least this line
   :private-members:
   :show-inheritance:                           <-- plus I want to show inheritance...
   :inherited-members:                          <-- ...and inherited members too

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

   .. rubric:: {{ _('Methods definition') }}
