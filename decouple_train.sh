for net in ResNet18
do
	for mode in classifier
	do
		for epochs in 120
		do
		       for beta in 6
		       do 	       
			python decouple_train.py --beta $beta --net $net --epochs $epochs --mode $mode | tee log/decouple_training/beta_$beta\_$mode\_$epochs\_$net.txt
		done
		done
	done
done
