for beta in 5.0
do
	for net in SmallCNN
	do
		for epoch in 100
		do
			for gamma in 10000
			do 
	python amart_train.py --gamma $gamma --epochs $epoch --net $net  --beta $beta --epsilon 0.3 | tee ../log/amart_training/mnist/$net/change_gamma_100\_beta_$beta\_0.3.txt
done
done
done
done
