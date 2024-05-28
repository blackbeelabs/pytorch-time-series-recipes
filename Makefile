PYENV_VERSION=3.10.13
PROJECT_NAME=pytorch-time-series-recipes
PYTHONPATH = $(HOME)/Developer/${PROJECT_NAME}/src


setup_env:
	pyenv virtualenv ${PYENV_VERSION} ${PROJECT_NAME}
	pyenv local ${PROJECT_NAME}
	pip install -r requirements.txt
	pip install -r requirements-jupyter.txt

train:
	@echo ${PYTHONPATH}
	python ${PYTHONPATH}/pipelines/train.py

teardown_env:
	# pyenv deactivate ${PROJECT_NAME}
	pyenv uninstall ${PROJECT_NAME}

qa_data_loader:
	python ${PYTHONPATH}/pipelines/qa/qa_data_loader.py