for beta in 6 
do
	for net in ResNet18
	do
	python bn_train.py --net $net  --beta $beta --epsilon 0.03137 | tee log/bn_training/mart_beta_$beta\_$net\_0.03137.txt
done
done
