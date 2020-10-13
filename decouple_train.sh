for net in Decouple18
do
		for epochs in 120
		do
		       for beta in 6
		       do
		for class in mart
		do 		
			python decouple_train.py --classify-loss $class  --beta $beta --net $net --epochs $epochs  | tee log/decouple_training/classify_$class\_beta_$beta\_$epochs\_$net.txt
		done
		done
	done
done
