for net in ResNet18
do 
for max_loss in CE
do 
	for min_loss in CE
	do
	for attack in PGD
	do 
		for eps in 0.03137
		do
#			tensorboard --logdir runs/$min_loss\_$max_loss\_$net/adv_$attack\_$eps
		    	python adv_train.py --resume  --net $net --min_loss $min_loss --max_loss $max_loss --attack_method $attack --epsilon $eps | tee log/adv_training/$attack\_$eps\_$min_loss\_$max_loss\_$net.txt
done
done
done
done
done
