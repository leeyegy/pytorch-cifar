for beta in 6.0
do
	for net in WideResNet
	do
		for epoch in 50
		do
	python mart_train.py --epochs $epoch --net $net  --beta $beta --epsilon 0.03137 | tee log/mart_training/whole_234567_$epoch\_beta_$beta\_$net\_0.03137.txt
done
done
done
