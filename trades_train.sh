for net in Decouple18
do
	for beta in 6.0
	do
		for gamma in 10.0
		do
	python trades_train.py --resume_best --gamma $gamma  --net $net --apply_FN --beta $beta --epsilon 0.03137 | tee log/trades_training/resume_FN_feature_gamma_$gamma\_beta_$beta\_$net\_0.03137.txt
done
done
done 
