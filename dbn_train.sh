for net in ResNet18_dbn
do 
		for eps in 0.03137
		do
			for beta in 6
			do
    python dbn_train.py --beta $beta --net $net --epsilon $eps | tee log/trades_training/beta_$beta\_$eps\_$net.txt
done
done
done
