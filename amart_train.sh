for beta in 3
do
	for net in ResNet18
	do
		for loss in amart
		do
			for gamma in 0.8
			do
	python amart_train.py --gamma $gamma --loss $loss  --net $net  --beta $beta --epsilon 0.03137 | tee log/amart_training/$loss\_beta_$beta\_$net\_gamma_$gamma\_0.03137.txt
done
done
done
done
