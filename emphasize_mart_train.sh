for beta in 6.0
do
	for net in WideResNet
	do
		for king in 0 1 2 3 4 5 6 7 8 9
		do
	python emphasize_mart_train.py --emphasize-label $king --net $net  --beta $beta --epsilon 0.03137 | tee log/emphasize_mart_training/king_$king\_beta_$beta\_$net\_0.03137.txt
done
done
done
