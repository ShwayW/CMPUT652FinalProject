for i in {3..3}; do
	python processPath.py "${i}"
	python mergeLevelAndpath.py "${i}"
	python pathToSnake.py "${i}"
	python visualize.py "${i}"
done