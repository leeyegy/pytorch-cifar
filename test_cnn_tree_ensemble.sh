for beta in 0.3 0.8
do
	python test_cnn_tree_ensemble_attack.py --beta $beta | tee log/cnn_tree_training/$beta.txt
done 
