To build the notebook with python:

1. Install the dependencies:

   ```shell
   # Install uv if it's not already on your system.
   python3 -m pipx install uv
   # Do not let uv create a virtual environment in the current
   # directory, because that slows down the AI agent harness.
   export UV_PROJECT_ENVIRONMENT=$HOME/.local/share/venv-stanodeffpy
   # Create the virtual env from the pyproject.toml.
   uv sync
   # Only if you're planning to commit changes to the notebook.
   uv sync --group dev
   uv run nbstripout --install
   ```

1. Build the specific notebook and export to see the output Markdown, HTML,
   PDF, and Python files, for example:

   ```shell
   uv run jupyter execute 01-stan-ode-forcing-function.ipynb
   uv run jupyter nbconvert --to markdown 01-stan-ode-forcing-function.ipynb
   uv run jupyter nbconvert --to html 01-stan-ode-forcing-function.ipynb
   uv run jupyter nbconvert --to pdf 01-stan-ode-forcing-function.ipynb
   uv run jupyter nbconvert --to python 01-stan-ode-forcing-function.ipynb
   ```
