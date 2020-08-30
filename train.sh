for net in DPN92
do 
for loss in CE
do 
    python main.py --net $net --loss $loss | tee log/$loss\_$net.txt
done
done
