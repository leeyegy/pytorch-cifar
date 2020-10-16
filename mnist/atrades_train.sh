for beta in 10.0
do
	for net in SmallCNN
	do
		for epoch in 100
		do
			for gamma in 1
			do 
	python atrades_train.py --gamma $gamma --epochs $epoch --net $net  --beta $beta --epsilon 0.3 | tee ../log/atrades_training/mnist/$net/gamma_$gamma\_beta_$beta\_0.3.txt
done
done
done
done
