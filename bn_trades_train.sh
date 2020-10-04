for beta in 6 
do
	for epoch in 120
	do 
	for net in ResNet18
	do
	python bn_trades_train.py --epoch_nb $epoch  --net $net  --beta $beta --epsilon 0.03137 | tee log/bn_training/120_trades_beta_$beta\_$net\_0.03137.txt
done
done
done 
