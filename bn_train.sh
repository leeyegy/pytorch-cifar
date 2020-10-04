for beta in 6 
do
	for epoch in 50
	do 
	for net in ResNet18
	do
	python bn_train.py --epoch_nb $epoch --random_bn --net $net  --beta $beta --epsilon 0.03137 | tee log/bn_training/50_mart_random_bn_beta_$beta\_$net\_0.03137.txt
done
done
done 
