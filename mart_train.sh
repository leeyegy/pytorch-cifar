for beta in 5.0
do
	for net in HBPNet
	do
		for epoch in 120
		do
	python mart_train.py --epochs $epoch --net $net  --beta $beta --epsilon 0.03137 | tee log/mart_training/$net/beta_$beta\_0.03137.txt
done
done
done
