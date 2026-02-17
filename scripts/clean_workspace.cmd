@echo off
setlocal

if exist out rmdir /s /q out
if exist __pycache__ rmdir /s /q __pycache__
if exist analysis\__pycache__ rmdir /s /q analysis\__pycache__
if exist benchmarks\__pycache__ rmdir /s /q benchmarks\__pycache__
if exist evaluators\__pycache__ rmdir /s /q evaluators\__pycache__
if exist geometry\__pycache__ rmdir /s /q geometry\__pycache__
if exist optimization\__pycache__ rmdir /s /q optimization\__pycache__
if exist docs\class-diagram.png.crdownload del /q docs\class-diagram.png.crdownload

echo Cleanup complete.
endlocal
