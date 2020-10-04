for beta in 5
do
	for net in ResNet18
	do
		for loss in mart-topk
		do
			for ratio in 0.95
			do
	python ohem_train.py --ratio $ratio --loss $loss  --net $net  --beta $beta --epsilon 0.03137 | tee log/ohem_training/$loss\_beta_$beta\_$net\_ratio_$ratio\_0.03137.txt
done
done
done
done
