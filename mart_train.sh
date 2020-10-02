for beta in 6 
do
	for net in ResNet18
	do
	python mart_train.py --net $net  --beta $beta --epsilon 0.03137 | tee log/mart_training/beta_$beta\_$net\_0.03137.txt
done
done
