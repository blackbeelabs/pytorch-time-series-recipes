PYENV_VERSION=3.10.13
PROJECT_NAME=pytorch-time-series-recipes
PYTHONPATH = $(HOME)/Developer/${PROJECT_NAME}/src


setup_env:
	pyenv virtualenv ${PYENV_VERSION} ${PROJECT_NAME}
	pyenv local ${PROJECT_NAME}
	pip install -r requirements.txt
	pip install -r requirements-jupyter.txt

hello_world:
	python ${PYTHONPATH}/hello_world.py

teardown_env:
	# pyenv deactivate ${PROJECT_NAME}
	pyenv uninstall ${PROJECT_NAME}

