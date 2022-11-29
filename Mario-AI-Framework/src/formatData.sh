for i in {1..1}; do
	python processPath.py "${i}"
	python mergeLevelAndpath.py "${i}"
	python pathToSnake.py "${i}"
done