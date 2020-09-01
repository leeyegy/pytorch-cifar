for net in ResNet18
do 
for loss in CS
do 
    python main.py --resume --net $net --loss $loss | tee log/$loss\_$net.txt
done
done
