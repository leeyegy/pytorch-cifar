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
		   python init_train.py --lr 0.001 --init checkpoint/clean_ce_res18.pth  --net $net --min_loss $min_loss --max_loss $max_loss --attack_method $attack --epsilon $eps | tee log/adv_training/init_$attack\_$eps\_$min_loss\_$gamma\_$max_loss\_$net.txt
done
done
done
done
done
