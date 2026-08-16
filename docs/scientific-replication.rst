Scientific replication
======================

.. meta::
   :description: Distinguish first use of factorlasso from the optional JSS and sign-pooling
      manuscript replication workflows and their additional dependencies.

Package evaluation and manuscript replication are separate workflows. To evaluate the library,
install ``factorlasso`` and complete :doc:`getting-started` and :doc:`task-guides`. Those paths are
offline, use the supported package API, and need neither a paper checkout nor competitor packages.

The manuscript trees preserve frozen scientific inputs, seeds, exhibit generators, and additional
toolchains. Their review status is not a software stability claim, and reproducing them is not a
prerequisite for using the package.

JSS software-paper replication
------------------------------

The `JSS replication tree
<https://github.com/ArturSepp/factorlasso/tree/main/papers/jss_2026>`_ accompanies
*factorlasso: Hierarchical Clustering Group LASSO (HCGL) with Cluster-Pooled Sign Derivation for
Multi-Asset Factor Models in Python*. Its committed 2026-06 ETF and factor panels are the canonical
inputs, so current network data are not part of the reproduced result.

From ``papers/jss_2026`` in a source checkout:

.. code-block:: console

   python -m pip install -r requirements.txt
   python replicate.py --log replication_output.txt

The default is the quick smoke path. ``python replicate.py --full`` uses the full manuscript seed
set and is the command for exact manuscript numbers; it is materially longer. The comparison stage
also needs the compiled or platform-specific packages named in that tree's README. Install failures
for those research dependencies do not imply that the core ``factorlasso`` wheel is broken.

Sign-pooling replication
------------------------

The `sign-pooling replication tree
<https://github.com/ArturSepp/factorlasso/tree/main/papers/sign_pooling_2026>`_ accompanies
*Gated Cluster-Pooled Sign Constraints for Multi-Output Sparse Regression*. It includes the public
yeast eQTL inputs and cached result tables. The archived research bundle has DOI
`10.5281/zenodo.21000294 <https://doi.org/10.5281/zenodo.21000294>`_.

From ``papers/sign_pooling_2026`` in a source checkout:

.. code-block:: console

   python -m pip install -r replication/requirements.txt
   make sims
   make eqtl

``make paper`` additionally needs the Elsevier CAS LaTeX template from CTAN. The replication pins
the historical FactorLasso version used by that manuscript; do not silently substitute the newest
package and call changed output a reproduction.

Evidence and provenance
-----------------------

Keep the generated session log with any reproduced exhibits. Record the source commit, package and
solver versions, Python version, platform, command, and whether the quick or full path was used.
Never overwrite committed tables merely to make them agree with a changed environment: report the
difference and establish whether it comes from code, dependency resolution, or platform-specific
solver tolerance.

Use :doc:`interoperability` and the `compatibility policy
<https://github.com/ArturSepp/factorlasso/blob/main/COMPATIBILITY.md>`_ for software contracts. Use
the paper-specific README, archived DOI where applicable, and session log for scientific
reproducibility.
