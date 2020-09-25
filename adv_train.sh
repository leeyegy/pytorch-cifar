for net in ResNet18
do 
for max_loss in CE
do 
	for min_loss in Easy2hardLoss
	do
	for attack in PGD
	do 
		for eps in 0.03137
		do
			for gamma in 0.5  
			do
		    	python adv_train.py --gamma $gamma  --net $net --min_loss $min_loss --max_loss $max_loss --attack_method $attack --epsilon $eps | tee log/adv_training/$attack\_$eps\_$min_loss\_$gamma\_$max_loss\_$net.txt
done
done
done
done
done
done
