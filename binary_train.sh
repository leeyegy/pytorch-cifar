for net in ResNet18
do 
for max_loss in CE
do 
	for min_loss in BalanceLoss
	do
	for attack in PGD
	do 
		for eps in 0.03137
		do
			for pick in 3
			do
		    	python binary_train.py --pick_up $pick  --net $net --min_loss $min_loss --max_loss $max_loss --attack_method $attack --epsilon $eps | tee log/adv_training/pick_$pick\_$attack\_$eps\_$min_loss\_$max_loss\_$net.txt
done
done
done
done
done
done
