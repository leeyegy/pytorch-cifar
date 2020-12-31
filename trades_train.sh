for net in Decouple18
do
	for beta in 6.0
	do
		for gamma in 1.0
		do
	python trades_train.py --gamma $gamma  --net $net --beta $beta --epsilon 0.03137 | tee log/trades_training/no_FN_feature_gamma_$gamma\_beta_$beta\_$net\_0.03137.txt
done
done
done 
