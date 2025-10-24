# Setting Up the Repository

Follow these steps to set up the repository:

## Prepare Environment
### Install Rust Language Systems
Follow [link](https://rustup.rs) to install the Rust language systems.

### Install Python Project Manager
Install **uv**, a python project & package manager, following [instruction](https://docs.astral.sh/uv/getting-started/installation/).

### Configure Git Credential
1. Generate an Access Token in GitHub
1. Create a Credential File

    Add the following string to `~/.git-credentials`. (Replace `[username]` and `[access token]` with your information.)
    ```
    https://[username]:[access token]@github.com
    ```

1. Setting Credential Helper
    ```bash
    git config --global credential.helper 'store'
    ```

## Set up Repository
### Clone Repository
```bash
git clone https://github.com/shinomi-lab/xa-promote.git
cd xa-promote
```

### Build Simulation App  
```bash
cargo build --release
```
`./target/relase/xa-promote` application will be built.

### Set up Data Analysis Environment
The directory `./analysis` is a python project for execute jupyter notebook files (`.ipynb`) in VS Code.

1. Open Workspace in Visual Studio Code

    Open `./xa-promote.code-workspace` via "Open Workspace from File..." menu.

1. Install Dependencies
    ```bash
    cd ./analysis
    uv sync
    ```
    Then, `.venv` directory will be created in the **analysis** project.

When you open or create a notebook file, choose the Python kernel of `./analysis/.venv` in VS Code.

# Implemented Algorithm
## Brute-force search
1. Perform Algorithm
    ```bash
    ./target/release/xa-promote blute-force [argments...]
    ```
    The usage instructions can be displayed using the `--help` option.

1. Analyze Experimental Result

    Check `./analysis/sample.ipynb` file.


# Directory Structure
- core
    - Implementation of models
- runner
    - Implementation of search algorithms
- util
    - General tools for core and runner
- src
    - Simulation application
- test
    - Some data for testing
- analysis
    - Python project for data analysis
