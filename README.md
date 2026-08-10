# Using forcing functions in ordinary differential equations (ODE) in Stan

This GitHub pages website
explains how to use forcing functions in ODEs
in Stan as well as lower-level ODE solvers.

<https://immusystems-lab.github.io/stan-ode-forcing-function>

Note this repository is assembling a *collection* of notebooks
for the reasons explained in the next section.

## Experiments on AI-assisted ODE forcing function code generation

Experiments are planned to understand
how to better write notebooks on niche scientific computing topics that are
accessible to both human readers as well as AI coding agents
around the following dimensions:

1. Notebook outputs formats: HTML, PDF, Markdown.
1. Notebook pagination: All-in-one, separate pages.
1. Impact of using a domain-specific language (DSL): Stan, non-DSL ODE solver.
1. Computing languages: R (RStan), Python (PyStan).
1. Code editor AI agents: [those supported by
   the Posit Positron and RStudio code editors.]
1. Web browser AI agents: Claude Haiku, Claude Sonnet, Claude Opus
   (access provided by the University of Pittsburgh).

Evaluation criteria:

1. Correctness: compilation success, runtime behavior, numerical agreement
1. Prompt efficiency: number of tokens used.
1. Prompt complexity: how detailed the prompt needs to be
   to reach output of varying quality.

Outcomes of interest:

1. General guidance for scientific notebook documentation authors
   on how to describe scientific computing concepts
   for human readers, AI agents,
   and finding a balancing between both.
1. Relationship of AI agent architecture with prompt complexity
   for this scientific computing code.
1. Ranking of AI agents for this scientific computing problem.
1. Prompt complexity when converting code from a non-DSL to a DSL language
   compared to directly writing in the DSL language.
1. Output quality differences between Python and R;
   Python output is expected to be of higher quality due to wider use
   in this area.
1. Observable context size limitations and minimizing token usage
   by splitting notebooks across chapters.
1. Types of introductory content in notebooks that reduce prompt complexity
   while still producing accepted reference output.
1. Protocol for reproducible outcomes from prompt to output.
1. Complete prompt logs with output.

<img src="https://us-rse.org/usrse26/assets/img/usrse26-round-logo.svg" width="200" align="right"/>

## Accepted abstract submission to USRSE'26

Title: Beyond human readers:
designing domain-specific language notebooks for
AI-assisted code generation

Authors:
Pariksheet Nanda, Rocco John Caprara[^presenter], Jason Edward Shoemaker

Abstract (300 words max):

Ordinary differential equations (ODEs) are widely used for
mathematical modeling of natural systems.  A less common but critical
ODE solver input is the incorporation of fixed temporal data
representing a process whose underlying equation is unknown, called a
forcing function.  Forcing functions are more readily supported in
lower-level programming languages than in higher-level,
domain-specific programming languages (DSLs) that allow fewer
mathematical primitives and are more complex to extend.  In the spirit
of this year's USE-RSE conference theme, "Advancing Science in the Age
of AI", we propose a higher standard for technical documentation: a
notebook designed to explain forcing functions in the Stan DSL that
serves both human learners and AI coding agents.  We evaluate how well
AI agents known to write Stan code can incorporate forcing functions
by testing the effects of varied prompts and notebook variants that
focus on AI-suitable explanations and reference links.  We use the AI
assistants built into the Positron and RStudio code editors from Posit
to assist with the development of Stan code and compare suggestions
for RStan and PyStan projects.  Notebook formats tested include raw
Markdown, HTML output with minimal JavaScript, and PDF output.  As a
baseline, we also compare AI-assisted forcing function code against
implementations using non-DSL ODE solvers.  Our findings yield a
ranked set of strategies for teaching AI coding agents new and
challenging concepts in DSLs, along with three notebook variants
optimized for: (1) human readers, (2) AI agents, and (3) our best
balance for both.

Keywords (3 max):

- retrieval-augmented generation
- domain specific language
- mathematical modeling

[^presenter]: Presenter
