@echo off
echo Opening Lab3 Jupyter Notebooks in JupyterLab...
echo.

start "" jupyter lab "Lab3_Jupyter.ipynb"
timeout /t 3 /nobreak >nul
start "" jupyter lab "Lab3_Jupyter_Solved.ipynb"

echo Notebooks are opening in JupyterLab.
pause
