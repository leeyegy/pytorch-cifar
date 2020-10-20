for beta in 5.0
do
	for net in ResNet18_aux
	do
		for epoch in 120
		do
	python mart_aux_train.py  --resume_best --epochs $epoch --net $net  --beta $beta --epsilon 0.03137 | tee log/mart_training/$net/shuffle_4_gamma_1_beta_$beta\_0.03137.txt
done
done
done
