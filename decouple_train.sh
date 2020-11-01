for net in Decouple18
do
		for epochs in 100
		do
		       for beta in 15
		       do
			for norm_weight in normalize 
			do
			python decouple_train.py --norm_weight $norm_weight --beta $beta --net $net --epochs $epochs  | tee log/decouple_training/$norm_weight\_blend_1_mart_5_weight\_beta_$beta\_$epochs\_$net.txt
		done
		done
	done
	done
