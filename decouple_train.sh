for net in Decouple18
do
		for epochs in 120
		do
		       for beta in 6
		       do 	       
			python decouple_train.py --beta $beta --net $net --epochs $epochs  | tee log/decouple_training/beta_$beta\_$epochs\_$net.txt
		done
		done
	done
