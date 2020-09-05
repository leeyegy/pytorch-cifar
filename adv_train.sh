for net in ResNet18
do 
for loss in CE
do 
	for attack in PGD
	do 
		for eps in 0.03137
		do
    python adv_train.py   --net $net --loss $loss --attack_method $attack --epsilon $eps | tee log/adv_training/$attack\_$eps\_$loss\_$net.txt
done
done
done
done
