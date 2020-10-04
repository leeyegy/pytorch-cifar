for beta in 6.0
do
	for net in ResNet18
	do
		for epoch in 55
		do
	python mart_train.py --epochs $epoch --net $net  --beta $beta --epsilon 0.03137 | tee log/mart_training/$epoch\_beta_$beta\_$net\_0.03137.txt
done
done
done
