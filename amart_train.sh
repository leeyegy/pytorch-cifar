for beta in 4
do
	python amart_train.py --beta $beta --epsilon 0.03137 | tee log/amart_training/beta_$beta\_0.03137.txt
done
