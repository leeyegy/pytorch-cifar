for beta in 5.0
do
	for net in ResNet18_aux
	do
		for epoch in 100
		do
	python mart_aux_train.py  --epochs $epoch --net $net  --beta $beta --epsilon 0.03137 | tee log/mart_training/$net/gamma_1_beta_$beta\_0.03137.txt
done
done
done
