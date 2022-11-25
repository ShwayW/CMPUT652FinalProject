for index in {1..15}; do
	python mergeLevelAndPath.py "${index}"
	python visualize.py "${index}"
	python pathToSnake.py "${index}" 
done