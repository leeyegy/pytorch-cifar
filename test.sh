for attack in FGSM PGD 
do 
for epsilon in 0.01569
do 
for loss in CS
do 
for test_model_path in checkpoint/CS_ResNet18/adv_PGD_0.03137/ckpt.pth
do
	python test_attack.py --attack_method $attack --loss $loss  --epsilon $epsilon --test_model_path $test_model_path | tee log/test_resnet18/CS_$attack\_$epsilon.txt
done
done
done 
done 
