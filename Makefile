PYENV_VERSION=3.10.13
PROJECT_NAME=pytorch-time-series-recipes

hello_world:
	@echo "Hello World"

setup_pyenv:
	pyenv virtualenv ${PYENV_VERSION} ${PROJECT_NAME}

install_pyenv:
	pyenv local ${PROJECT_NAME}
	pip install -r requirements.txt
	pip install -r requirements-jupyter.txt

setup_env: setup_pyenv install_pyenv
teardown_env:
	# pyenv deactivate ${PROJECT_NAME}
	pyenv uninstall ${PROJECT_NAME}