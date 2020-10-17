for beta in 5.0
do
	for net in ResNet18
	do
		for epoch in 100
		do
			for low_mode in low_12
			do
	python mart_train.py --low_mode $low_mode  --epochs $epoch --net $net  --beta $beta --epsilon 0.03137 | tee log/mart_training/$net/$low_mode\_beta_$beta\_0.03137.txt
done
done
done
done
