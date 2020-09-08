for net in ResNet18
do 
for loss in CE
do 
    python main.py --val_only --resume  --net $net --loss $loss 
done
done
