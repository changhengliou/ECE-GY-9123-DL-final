for i in {1..4}
do 
	python3 sumbasic.py simplified ./docs/doc$i-*.txt > ./outputs/simplified-$i.txt;
	python3 sumbasic.py orig ./docs/doc$i-*.txt > ./outputs/orig-$i.txt; 
	python3 sumbasic.py leading ./docs/doc$i-*.txt > ./outputs/leading-$i.txt; 
done