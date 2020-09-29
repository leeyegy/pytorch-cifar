for gamma in 1.0
do
	for net in ResNet18
	do
		for beta in 6.0
		do
	python atrades_train.py  --gamma $gamma --net $net  --beta $beta --epsilon 0.03137 | tee log/atrades_training/beta_$beta\_gamma_$gamma\_$net\_0.03137.txt
done
done
done
