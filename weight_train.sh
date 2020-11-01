for net in Tanh18
do
		for epochs in 100
		do
		       for beta in 100
		       do
			for s in 0.1
			do
			python weight_train.py --s $s --beta $beta --net $net --epochs $epochs  | tee log/weight_training/s_$s\_weight\_beta_$beta\_$epochs\_$net.txt
		done
		done
	done
	done
