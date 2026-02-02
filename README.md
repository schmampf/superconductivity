# install superconductivity.api as sc
python -m pip install -e .

# activate virtual environment
source ~/Documents/.venv/bin/activate
deactivate

git rm -r --cached .
git add .
git commit -m "Apply .gitignore (remove tracked ignored files)"