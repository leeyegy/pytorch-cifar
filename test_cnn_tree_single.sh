for beta in 0.8 1 
do
	python test_cnn_tree_attack.py --beta $beta | tee log/cnn_tree_training/50epoch_whole_234567_single_$beta.txt
done 
