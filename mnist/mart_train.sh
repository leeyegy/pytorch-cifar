for beta in 5.0
do
	for net in SmallCNN
	do
		for epoch in 100
		do
	python mart_train.py --epochs $epoch --net $net  --beta $beta --epsilon 0.3 | tee ../log/mart_training/mnist/$net/beta_$beta\_0.3.txt
done
done
done
