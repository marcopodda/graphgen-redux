export CUDA_VISIBLE_DEVICES=1
python manage.py train --dataset-name Lung --gpu 0 --epochs 550
python manage.py train --dataset-name Yeast --gpu 0 --epochs 250
python manage.py train --dataset-name All --gpu 0 --epochs 350
python manage.py train --dataset-name ENZYMES --gpu 0 --epochs 4000
python manage.py train --dataset-name citeseer --gpu 0 --epochs 400
python manage.py train --dataset-name cora --gpu 0 --epochs 400

# python manage.py generate --exp-path RESULTS/All/graphgen-redux 
# python manage.py generate --exp-path RESULTS/Yeast/graphgen-redux
# python manage.py generate --exp-path RESULTS/Lung/graphgen-redux
# python manage.py generate --exp-path RESULTS/ENZYMES/graphgen-redux
# python manage.py generate --exp-path RESULTS/citeseer/graphgen-redux
# python manage.py generate --exp-path RESULTS/cora/graphgen-redux

# python manage.py evaluate --exp-path RESULTS/All/graphgen-redux 
# python manage.py evaluate --exp-path RESULTS/Yeast/graphgen-redux 
# python manage.py evaluate --exp-path RESULTS/Lung/graphgen-redux 
# python manage.py evaluate --exp-path RESULTS/ENZYMES/graphgen-redux
# python manage.py evaluate --exp-path RESULTS/citeseer/graphgen-redux 
# python manage.py evaluate --exp-path RESULTS/cora/graphgen-redux

