for beta in 0 0.3 0.5 0.8 1 
do
	for net in WideResNet
	do
	python test_cnn_tree_attack.py --net $net  --beta $beta | tee log/cnn_tree_training/$net\_50epoch_whole_234567_single_$beta.txt
done 
done
