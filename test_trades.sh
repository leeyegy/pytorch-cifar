for attack in FGSM PGD 
do 
for epsilon in 0.00784 0.03137 0.06275
do 
for loss in CE
do 
for test_model_path in checkpoint/trades-res18-epoch120.pt
do
	python test_attack.py --attack_method $attack --loss $loss  --epsilon $epsilon --test_model_path $test_model_path | tee log/test_resnet18/Trades_$attack\_$epsilon.txt
done
done
done 
done 
