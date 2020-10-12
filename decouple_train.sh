for net in ResNet18
do
	for mode in classifier
	do
		for epochs in 120
		do 
			python decouple_train.py --net $net --epochs $epochs --mode $mode | tee log/decouple_training/$mode\_$epochs\_$net.txt
		done
	done
done
