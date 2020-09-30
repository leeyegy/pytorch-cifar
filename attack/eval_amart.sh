for epsilon  in 0.03137
do
	for step in 20
	do 
		python eval_MDAttack.py --num-steps $step --epsilon $epsilon --test_model_path ../checkpoint/amart_ResNet18/beta_6/ckpt_last.pth --md | tee amart_6_last.txt 
	done
done
