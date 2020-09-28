for beta in 5
do
	for net in WideResNet
	do
	python amart_train.py --net $net  --beta $beta --epsilon 0.03137 | tee log/amart_training/beta_$beta\_$net\_0.03137.txt
done
done
