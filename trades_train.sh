for net in ResNet18
do
	for beta in 6.0
	do
	python trades_train.py  --net $net  --beta $beta --epsilon 0.03137 | tee log/trades_training/beta_$beta\_$net\_0.03137.txt
done
done
