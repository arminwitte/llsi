#
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade isort black coverage pytest build twine

# isort
isort src/

# black
black src/

# pytest with coverage
coverage run -m pytest

# build wheel
rm -rf dist/
python3 -m build

# upload to pypi
python3 -m twine dist/*


