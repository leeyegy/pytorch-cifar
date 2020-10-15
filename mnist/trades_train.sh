for net in SmallCNN
do
	for beta in 5.0
	do
	python trades_train.py  --net $net  --beta $beta | tee ../log/trades_training/mnist/beta_$beta\_$net\_0.3.txt
done
done
