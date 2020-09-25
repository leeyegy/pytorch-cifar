#margin should be non-negative
for net in ResNet18
do 
for attack in PGD
do 
for eps in 0.03137
do
for margin_adv_anchor in 0.0
do
for margin_adv_most_confusing in 0.03 0.08 
do
    python margin_train.py  --net $net --margin_adv_anchor $margin_adv_anchor --margin_adv_most_confusing $margin_adv_most_confusing --attack_method $attack --epsilon $eps | tee log/margin_training/$margin_adv_anchor\_$margin_adv_most_confusing\_$attack\_$eps\_$net.txt
done
done
done
done
done
